#!/bin/bash

# 设置环境变量
export HF_HOME="/mnt/data4T-2/xhj/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0,1  # 根据你的GPU情况调整

# 本地模型路径
MODEL_PATH="/mnt/data4T-2/xhj/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"

# RefCOCO系列任务
TASKS="refcoco_bbox_val,refcoco_bbox_testA,refcoco_bbox_testB,refcoco_seg_val,refcoco_seg_testA,refcoco_seg_testB"

echo "Running Qwen2.5-VL-3B-Instruct on RefCOCO tasks..."
echo "Model path: $MODEL_PATH"
echo "Tasks: $TASKS"

# 运行评估
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=${MODEL_PATH},max_pixels=802816,attn_implementation=flash_attention_2 \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen25vl_3b_refcoco \
    --output_path ./outputs/

uv run python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="/mnt/data4T-2/xhj/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",max_pixels=12845056,attn_implementation=flash_attention_2 \
    --tasks refcoco \
    --batch_size 256