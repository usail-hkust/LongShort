# Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning ([PDF](https://arxiv.org/pdf/2505.11827))

<p align="center">

![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-blue)

</p>

<p align="center">

| **[1 Introduction](##introduction)** 
| **[2 Requirements](##requirements)**
| **[3 Usage](##usage)**
| **[4 Citation](##citation)**

</p>

## 1 Introduction
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/usail-hkust/LongShort/blob/main/figure/fig1.png">
</div>

Official code for paper "[Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2505.11827)".

Long⊗Short is an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts.

## 2 Requirements

This project rely on CUDA 12.6. If you see errors related to segmentation faults, double check the version your system is running with nvcc --version.

To run this project, we first create a python 3.11 environment and install dependencies:

```
conda create -n python3.11 LongShort
source activate LongShort
```

Then, install vLLM and FlashAttention:

```
pip install vllm==0.7.2
pip install setuptools && pip install flash-attn --no-build-isolation
```

Then, you can install the remaining dependencies via requirements file:

```
pip install -r requirements.txt
```

As we will visualize our project on wandb, you can log into your accounts as follows:

```
wandb login
```

## 3 Usage

The training of Long⊗Short is divided into automatic LongCoT chunking, SFT cold-start, and multi-turn RL training process.

### Automatic LongCoT Chunking

To conduct LongCoT chunking, you need to set your LongCoT trajectories and the output results_dir:

```
bash ./LongCoT_chunking/block_generate.sh "$model_dir" "$dataset_dir" "$result_dir"
```

The model_dir is the directory of LLM you used for automatic chunking, and the dataset_dir is the JSONL file that satisfies the following format:

```
{"problem": "", "solution": "", "answer": "", "response": ""}
```
where response is the corresponding LongCoT response for the given problem, while solution and answer are the ground-truth.
We also release our [OpenMath-ThoughtChunk1.8K](https://huggingface.co/datasets/yasNing/OpenMath-ThoughtChunk1.8K) on Hugging Face, you can download from Hugging Face:

```
from huggingface_hub import snapshot_download

repo_id = "yasNing/OpenMath-ThoughtChunk1.8K" 
local_dir = "./data/LongCoT1.8K/OpenMath-ThoughtChunk1.8K"  
local_dir_use_symlinks = False  #
token = "YOUR_KEY"  # hugging face access token

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=local_dir_use_symlinks,
    token=token
)
```

### SFT Cold Start

We follow [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to build this project. Specifically, we use 4 NVIDIA-A100 GPUs for full parameters fine-tuning, and an example is as follows:

```
bash ./SFT_Cold_Start/Qwen-LongCoT-sft.sh
```

### Multi-Turn RL Training

We follow [open-r1](https://github.com/huggingface/open-r1) to build this project. For single-node training of LLM models across 8 GPUs. We first spin up the vLLM server to run on 1 GPU for offline LLM sampling, 1 GPU for online LLM sampling, and then use 6 GPUs for RL training. An example is as follows:

```
bash ./Multi_Turn_RL/multi_turn_RL_LongCoT.sh
```

## 4 Citation

If you find our work is useful for your research, please consider citing:

```
@article{ning2025not,
  title={Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning},
  author={Ning, Yansong and Li, Wei and Fang, Jun and Tan, Naiqiang and Liu, Hao},
  journal={arXiv preprint arXiv:2505.11827},
  year={2025}
}
```



