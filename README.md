# LLaMA with Wrapyfi

Wrapyfi enables distributing LLaMA (inference only) on multiple GPUs/machines, each with less than 16GB VRAM 

**currently distributes on two cards only using ZeroMQ. Will support flexible distribution soon!** 

**This approach has only been tested on 7B model for now, using Ubuntu 20.04 with two 1080 Tis. Testing 13B/30B models soon!**
**UPDATE: Tested on Two 3080 Tis as well!!!**

### How to?

0. Replace all instances of <YOUR_IP> and <YOUR CHECKPOINT DIRECTORY> before running the scripts
  
1. Download LLaMA weights using the official form below and install this wrapyfi-examples_llama inside conda or virtual env:

  ```
  git clone https://github.com/modular-ml/wrapyfi-examples_llama.git
  cd wrapyfi-examples_llama
  pip install -r requirements.txt
  pip install -e .
  ```

3. Install Wrapyfi with the same environment:

  ```
  git clone https://github.com/fabawi/wrapyfi.git
  cd wrapyfi
  pip install .[pyzmq]
  ```

4. Start the Wrapyfi ZeroMQ broker from within the Wrapyfi repo:

  ```
  cd wrapyfi/standalone 
  python zeromq_proxy_broker.py --comm_type pubsubpoll
  ```

5. Start the first instance of the Wrapyfi-wrapped LLaMA from within this repo and env (order is important, dont start wrapyfi_device_idx=0 before wrapyfi_device_idx=1):

  ```
  CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=1 torchrun --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR CHECKPOINT DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 1
  ```
6. Now start the second instance (within this repo and env) :

  ```
  CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=1 torchrun --master_port=29503 --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR CHECKPOINT DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 0
  ```

7. You will now see the output on both terminals

8. EXTRA: To run on different machines, the broker must be running on a specific IP in step 4. Start the ZeroMQ broker by setting the IP and provide the env variables for steps 5+6 e.g.,

  ```
  ### (replace 10.0.0.101 with <YOUR_IP> ###
  
  # step 4 modification 
  python zeromq_proxy_broker.py --socket_ip 10.0.0.101 --comm_type pubsubpoll
  
  # step 5 modification
  CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=1 WRAPYFI_ZEROMQ_SOCKET_IP='10.0.0.101' torchrun --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR CHECKPOINT DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 1
  
  # step 6 modification
  CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=1 WRAPYFI_ZEROMQ_SOCKET_IP='10.0.0.101' torchrun --master_port=29503 --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR CHECKPOINT DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 0
  ```


# LLaMA 

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

### Setup
In a conda env with pytorch / cuda available, run
```
pip install -r requirements.txt
```
Then in this repository
```
pip install -e .
```

### Download
Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

### Inference
The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

### FAQ
- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

### Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

### License
See the [LICENSE](LICENSE) file.
