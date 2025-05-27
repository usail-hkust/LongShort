#!/bin/bash
# This script activates the DiDiAgent environment

# 调用 Python 脚本并获取未被占用的端口号
FREE_PORT=$(python3 find_free_port.py)

# 输出分配的端口号（可选）
echo "Assigned port: ${FREE_PORT}"

# Activate the environment
conda activate your_env_name

llamafactory-cli train examples/train_full/qwen2.5_shortcot_full_sft.yaml

llamafactory-cli export examples/merge_lora/qwen2.5_shortcot_full_sft.yaml
