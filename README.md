# LLaMA with Wrapyfi

Wrapyfi enables distributing LLaMA (inference only) on multiple GPUs/machines, each with less than 16GB VRAM 


## Benchmarks

LLaMA 7B: input_sequence=1024, output_sequence=256, batch_size=32

* LLaMA 7B: 2 GPUs/ Same Machine -> 2 TITAN Xp 12 GB

| GPU ID \| Type    | CPU Mem. \| Power | GPU Mem. | WPS |
|-----------------|----------|----------|-----|
| 0 \| TITAN Xp 12GB | 1.3 GB \| 79 W / 250 W | 8.8 GB    | 14  |
| 1 \| TITAN Xp 12GB | 1.3 GB \| 79 W / 250 W | 9.1 GB   | 14  |

* LLaMA 7B: 4 GPUs/ Same Machine -> 2 TITAN Xp 12 GB, 2 TITAN X 12 GB

| GPU ID \| Type    | CPU Mem. \| Power| GPU Mem. | WPS |
|-----------------|----------|----------|-----|
| 0 \| TITAN Xp 12GB | 1.3 GB \| 79 W / 250 W | 5.6 GB | 12  |
| 1 \| TITAN Xp 12GB | 1.3 GB \| 63 W / 250 W | 5.6 GB | 12  |
| 2 \| TITAN X 12GB | 1.3 GB \| 89 W / 250 W | 5.5 GB  | 12  |
| 3 \| TITAN X 12GB | 1.3 GB \| 99 W / 250 W | 6.2 GB  | 12  |

**UPDATE 1: More benchmarks to follow on 1080 Ti, 3080 Ti with 8 (4x4) remote**

**UPDATE 2: Much faster than CPU offloading approaches, and uses about 9 GB VRAM on each card with batch size: 32**

## Example on running 2 GPUs with 7B:

0. Replace all occurances of <YOUR_IP> and <YOUR_CHECKPOINT_DIRECTORY> before running the scripts
  
1. Download LLaMA weights using the official form below and install this wrapyfi-examples_llama inside conda or virtual env:

  ```
  git clone https://github.com/modular-ml/wrapyfi-examples_llama.git
  cd wrapyfi-examples_llama
  pip install -r requirements.txt
  pip install -e .
  ```

3. Install Wrapyfi within the same environment:

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
  CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=1 torchrun --nproc_per_node 1 example.py --ckpt_dir <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 1 --wrapyfi_total_devices 2
  ```
6. Now start the second instance (within this repo and env) :

  ```
  CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=1 torchrun --master_port=29503 --nproc_per_node 1 example.py --ckpt_dir <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 0 --wrapyfi_total_devices 2
  ```

7. You will now see the output on both terminals

### Running 7B on two different machines

To run on different machines, the broker must be running on a specific IP in step 4. Start the ZeroMQ broker by setting the IP and provide the env variables for steps 5+6 e.g.,

  ```
  ### (replace 10.0.0.101 with <YOUR_IP> ###
  
  # step 4 modification 
  python zeromq_proxy_broker.py --socket_ip 10.0.0.101 --comm_type pubsubpoll
  
  # step 5 modification
  CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=1 WRAPYFI_ZEROMQ_SOCKET_IP='10.0.0.101' torchrun --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 1 --wrapyfi_total_devices 2
  
  # step 6 modification
  CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=1 WRAPYFI_ZEROMQ_SOCKET_IP='10.0.0.101' torchrun --master_port=29503 --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 0 --wrapyfi_total_devices 2
  ```
  
### Running 7B on 4 machines

To run the model on more machines, make sure that the number of layers (32 layers for 7B, 40 for 13 B, etc.) is divisible by `wrapyfi_total_devices`. To run on 4 machines. Make sure `CUDA_VISIBLE_DEVICES` is set to the correct GPU for each. Execute these commands in order:

  ```
  ### (replace 10.0.0.101 with <YOUR_IP> ###
  
  # step 4 modification 
  python zeromq_proxy_broker.py --socket_ip 10.0.0.101 --comm_type pubsubpoll
  
  # step 5 modification
  CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=1 WRAPYFI_ZEROMQ_SOCKET_IP='10.0.0.101' torchrun --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 3 --wrapyfi_total_devices 4
  
  # step 6 modification
  CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=1 WRAPYFI_ZEROMQ_SOCKET_IP='10.0.0.101' torchrun --master_port=29503 --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 2 --wrapyfi_total_devices 4
  
  # add step 7 (on machine or device 3)
  CUDA_VISIBLE_DEVICES="2" OMP_NUM_THREADS=1 WRAPYFI_ZEROMQ_SOCKET_IP='10.0.0.101' torchrun --master_port=29504 --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 1 --wrapyfi_total_devices 4
  
  # add step 8 (on machine or device 4)
  CUDA_VISIBLE_DEVICES="3" OMP_NUM_THREADS=1 WRAPYFI_ZEROMQ_SOCKET_IP='10.0.0.101' torchrun --master_port=29505 --nproc_per_node 1 example.py --ckpt_dir <YOUR CHECKPOINT DIRECTORY>/checkpoints/7B --tokenizer_path <YOUR_CHECKPOINT_DIRECTORY>/checkpoints/tokenizer.model --wrapyfi_device_idx 0 --wrapyfi_total_devices 4
  
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
