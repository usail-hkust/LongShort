#!/bin/bash
nohup bash .//offline_inference/serving_LongCoT.sh > serving_LongCoT.log 2>&1 &

conda activate your_env_name

export WANDB_API_KEY=your_key
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/ShortCoT/example.yaml
