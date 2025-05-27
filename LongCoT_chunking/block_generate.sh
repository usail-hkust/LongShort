#!/bin/bash
# Activate the environment

conda activate your_env_name

model_dir=$1
dataset_dir=$2 # json
result_dir=$3 # jsonl

python block_gnerate.py $model_dir $dataset_dir $result_dir
