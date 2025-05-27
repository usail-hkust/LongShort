# Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning ([PDF](https://arxiv.org/pdf/2505.11827))

<p align="center">

![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-blue)

</p>

<p align="center">

| **[1 Introduction](#introduction)** 
| **[2 Requirements](#requirements)**
| **[3 Usage](#usage)**
| **[4 Citation](#citation)**

</p>

## 1 Introduction
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/usail-hkust/LongShort/blob/main/figure/fig1.png">
</div>

Official code for paper "[Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2505.11827)".

Long⊗Short is an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts.

## 2. Requirements

This project rely on CUDA 12.6. If you see errors related to segmentation faults, double check the version your system is running with nvcc --version.

To run this project, we first create a python 3.11 environment and install dependencies:

```
conda create -n python3.11 LongShort
source activate LongShort
```

Then, install vLLM and FlashAttention:

```
pip install vllm==0.8.4
pip install setuptools && pip install flash-attn --no-build-isolation
```





