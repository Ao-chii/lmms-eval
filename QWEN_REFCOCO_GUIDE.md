# Qwen2.5-VL-3B-Instruct 在 lmms-eval 上测试 RefCOCO 系列任务指南

## 1. 环境准备

### 1.1 安装 lmms-eval
```bash
cd /mnt/data4T-2/xhj/lmms-eval

# 使用 uv 安装（推荐）
uv sync

# 或者使用 pip 安装
pip install -e .

# 安装 Qwen VL 相关依赖
pip install qwen-vl-utils
```

### 1.2 检查必要依赖
```bash
# 对于 RefCOCO 任务，需要 Java 8 来运行 pycocoeval
java -version

# 如果没有安装，可以用 conda 安装
conda install openjdk=8
```

## 2. 可用的 RefCOCO 任务

### RefCOCO 系列任务包括：
- **refcoco_bbox_val**: RefCOCO 验证集 (bbox 预测)
- **refcoco_bbox_testA**: RefCOCO testA (bbox 预测)
- **refcoco_bbox_testB**: RefCOCO testB (bbox 预测)
- **refcoco_seg_val**: RefCOCO 验证集 (分割)
- **refcoco_seg_testA**: RefCOCO testA (分割)
- **refcoco_seg_testB**: RefCOCO testB (分割)

### RefCOCO+ 系列：
- **refcoco+_bbox_val**: RefCOCO+ 验证集 (bbox 预测)
- **refcoco+_bbox_testA**: RefCOCO+ testA (bbox 预测)  
- **refcoco+_bbox_testB**: RefCOCO+ testB (bbox 预测)
- **refcoco+_seg_val**: RefCOCO+ 验证集 (分割)
- **refcoco+_seg_testA**: RefCOCO+ testA (分割)
- **refcoco+_seg_testB**: RefCOCO+ testB (分割)

### RefCOCOg 系列：
- **refcocog_bbox_val**: RefCOCOg 验证集 (bbox 预测)
- **refcocog_bbox_test**: RefCOCOg 测试集 (bbox 预测)
- **refcocog_seg_val**: RefCOCOg 验证集 (分割)
- **refcocog_seg_test**: RefCOCOg 测试集 (分割)

## 3. 运行命令

### 3.1 测试单个任务
```bash
# 设置环境变量
export HF_HOME="/mnt/data4T-2/xhj/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0

# 本地模型路径
MODEL_PATH="/mnt/data4T-2/xhj/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"

# 运行 RefCOCO 验证集
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=${MODEL_PATH},max_pixels=802816,attn_implementation=flash_attention_2 \
    --tasks refcoco_bbox_val \
    --batch_size 1 \
    --log_samples \
    --output_path ./outputs/
```

### 3.2 运行完整的 RefCOCO 系列评估
```bash
# 使用提供的脚本
./run_local_qwen25vl.sh
```

### 3.3 快速测试设置
```bash
# 使用测试脚本进行快速验证
./test_setup.sh
```

## 4. 模型参数说明

- **pretrained**: 本地模型路径
- **max_pixels**: 图像最大像素数，影响图像分辨率和显存使用
- **attn_implementation**: 注意力机制实现，推荐 "flash_attention_2"
- **batch_size**: 批大小，通常设为 1 以节省显存
- **limit**: 限制测试样本数量（用于快速测试）

## 5. 输出说明

评估结果将保存在 `./outputs/` 目录下，包括：
- 各项指标的分数 (BLEU, METEOR, ROUGE-L, CIDEr)
- 详细的样本日志（如果使用了 `--log_samples`）
- 汇总报告

## 6. 常见问题

### 6.1 内存不足
- 减少 `max_pixels` 参数
- 确保 `batch_size=1`
- 使用 `device_map=auto` 进行多GPU分布

### 6.2 max_pixels 参数错误
如果遇到 `TypeError: '>' not supported between instances of 'int' and 'str'` 错误，说明 `max_pixels` 参数格式不正确。
- ❌ 错误: `max_pixels=1024*28*28` (会被当作字符串)
- ✅ 正确: `max_pixels=802816` (计算后的数值)

### 6.3 依赖问题
```bash
pip install httpx==0.23.3
pip install protobuf==3.20
pip install numpy==1.26.4
pip install sentencepiece
```

## 7. 预期运行时间

- 单个验证集任务: 约30-60分钟 (取决于硬件)
- 完整RefCOCO系列: 约3-6小时
- 快速测试(5个样本): 约1-2分钟