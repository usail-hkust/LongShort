#!/bin/bash
# This script activates the DiDiAgent environment
# Activate the environment

conda activate your_env_name  # Replace with your environment name

# port 8921
export CUDA_VISIBLE_DEVICES=7
python local_api_ShortCoT.py
