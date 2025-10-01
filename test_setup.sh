#!/bin/bash

# 设置环境变量
export HF_HOME="/mnt/data4T-2/xhj/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0

# 本地模型路径
MODEL_PATH="/mnt/data4T-2/xhj/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"

echo "Testing Qwen2.5-VL-3B-Instruct setup with a small RefCOCO validation task..."
echo "Model path: $MODEL_PATH"

# 先测试一个小任务验证设置
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=${MODEL_PATH},max_pixels=802816,attn_implementation=flash_attention_2 \
    --tasks refcoco_bbox_val_lite \
    --batch_size 1 \
    --limit 5 \
    --log_samples \
    --output_path ./outputs/test/