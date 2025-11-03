#!/bin/bash
# merge_fast.sh

CHECKPOINT_DIR="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/RL/verl/checkpoints/verl/Qwen2.5-7B-Instruct_qwen25_7b_17k_mathhard-7bhard/global_step_160/actor"
OUTPUT_DIR="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/models/Qwen2.5-7B-Instruct_qwen25_7b_17k_mathhard-7bhard"
WORLD_SIZE=8

# 并行worker数量（建议设置为CPU核心数）
NUM_WORKERS=$(nproc)  # 自动获取CPU核心数
# 或者手动设置：NUM_WORKERS=16

echo "=================================================="
echo "🚀 VERL Fast Conversion Mode"
echo "=================================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "World Size: $WORLD_SIZE"
echo "Parallel Workers: $NUM_WORKERS"
echo "=================================================="

# 显示系统资源
echo "💻 System Resources:"
echo "  CPU cores: $(nproc)"
echo "  Memory:"
free -h | grep -E "Mem|Swap"
echo ""

# 设置环境变量优化
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 执行转换
time python -u /lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/RL/verl/checkpoints/merge_verl.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --world_size "$WORLD_SIZE" \
    --num_workers "$NUM_WORKERS"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ SUCCESS!"
    echo "=================================================="
    ls -lh "$OUTPUT_DIR"
else
    echo "❌ Failed!"
    exit 1
fi