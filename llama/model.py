# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
# Modified by Fares Abawi (fares.abawi@uni-hamburg.de)

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


def chunk_layers(n_layers, max_idx):
    # Yield successive n-sized chunks from lst
    for i in range(0, len(n_layers), max_idx):
        yield n_layers[i:i + max_idx]

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024
    wrapyfi_device_idx: int = 0
    wrapyfi_total_devices: int = 2


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(MiddlewareCommunicator, nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, enabled=True):
        #super().__init__()
        MiddlewareCommunicator.__init__(self)
        nn.Module.__init__(self)
        if enabled:
            self.n_heads = args.n_heads
            self.dim = args.dim
            self.head_dim = args.dim // args.n_heads
            self.attention = Attention(args)
            self.feed_forward = FeedForward(
                dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
            )
            self.layer_id = layer_id
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "TransformerBlock", "$topic_name", should_wait=True, listener_kwargs=dict(load_torch_device="$device"))
    def forward_wrapped(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], topic_name="/llama/transformer_block_x", device="cuda:0"):
        out = self(x,start_pos,freqs_cis,mask)
        return out,

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(MiddlewareCommunicator, nn.Module):
    def __init__(self, params: ModelArgs):
        MiddlewareCommunicator.__init__(self)
        nn.Module.__init__(self)

        self.params = params
        print("WRAPYFI_TOTAL_DEVICES",self.params.wrapyfi_total_devices)
        print("WRAPYFI_DEVICE_INDEX",self.params.wrapyfi_device_idx)
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        self.wrapyfi_layer_topics = list(range(params.n_layers))

        if self.params.wrapyfi_total_devices == -1:
            self.activate_communication(self.transformer_logits, None)
            device_layer_ranges = range(params.n_layers)
        elif self.params.wrapyfi_device_idx < self.params.wrapyfi_total_devices - 1:
            self.activate_communication(self.transformer_logits, "listen")
            device_layer_ranges = chunk_layers(range(params.n_layers), params.n_layers//self.params.wrapyfi_total_devices)
            device_layer_ranges = list(device_layer_ranges)[self.params.wrapyfi_device_idx]
        elif self.params.wrapyfi_device_idx == self.params.wrapyfi_total_devices - 1:
            self.activate_communication(self.transformer_logits, "publish")
            device_layer_ranges = chunk_layers(range(params.n_layers), params.n_layers//self.params.wrapyfi_total_devices)
            device_layer_ranges = list(device_layer_ranges)[self.params.wrapyfi_device_idx]

        for layer_id in range(params.n_layers):
            if self.params.wrapyfi_total_devices == -1:
                transformer_block = TransformerBlock(layer_id, params)
                transformer_block.activate_communication(transformer_block.forward_wrapped, None)

            elif layer_id == 0 and self.params.wrapyfi_device_idx == 0:
                transformer_block = TransformerBlock(layer_id, params)
                transformer_block.activate_communication(transformer_block.forward_wrapped, None)
            elif layer_id == params.n_layers - 1 and self.params.wrapyfi_device_idx == self.params.wrapyfi_total_devices - 1:
                transformer_block = TransformerBlock(layer_id, params)
                transformer_block.activate_communication(transformer_block.forward_wrapped, None)
            elif layer_id == min(device_layer_ranges):
                transformer_block = TransformerBlock(layer_id, params, enabled=False)
                transformer_block.activate_communication(transformer_block.forward_wrapped, "listen")
                self.wrapyfi_layer_topics[layer_id] -= 1
            elif layer_id == max(device_layer_ranges):
                transformer_block = TransformerBlock(layer_id, params)
                transformer_block.activate_communication(transformer_block.forward_wrapped, "publish")
            elif layer_id in device_layer_ranges:
                transformer_block = TransformerBlock(layer_id, params)
                transformer_block.activate_communication(transformer_block.forward_wrapped, None)
            else:
                transformer_block = TransformerBlock(layer_id, params, enabled=False)
                transformer_block.activate_communication(transformer_block.forward_wrapped, "disable")

            self.layers.append(transformer_block)

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        print("COMPLETED TRANSFORMER LOADING")

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "Transformer", "$topic_name", should_wait=True, listener_kwargs=dict(load_torch_device="$device"))
    def transformer_logits(self, x, topic_name="/llama/transformer_logits", device="cuda:0"):
        return x,


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer_id, layer in enumerate(self.layers):
            h, = layer.forward_wrapped(h, start_pos, freqs_cis, mask, topic_name=f"/llama/transformer_block_{self.wrapyfi_layer_topics[layer_id]}")
        if h is not None:
            h = self.norm(h)
            output = self.output(h[:, -1, :]).float()  # only compute last logits
        else:
                output = 0
        out_token, = self.transformer_logits(output)
        return out_token
