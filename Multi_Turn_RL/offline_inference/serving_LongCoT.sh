#!/bin/bash
# This script activates the DiDiAgent environment

# Activate the environment

conda activate your_env_name

# port 8920
export CUDA_VISIBLE_DEVICES=7
python local_api_LongCoT.py
