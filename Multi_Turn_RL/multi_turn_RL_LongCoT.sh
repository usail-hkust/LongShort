#!/bin/bash


nohup bash .//offline_inference/serving_ShortCoT.sh > serving_ShortCoT.log 2>&1 &

conda activate your_env_name

export WANDB_API_KEY=your_key

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=6 src/MTRL/grpo.py \
    --config recipes/LongCoT/example.yaml