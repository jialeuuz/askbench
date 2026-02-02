#!/bin/bash
# merge_fast.sh

CHECKPOINT_DIR="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/RL/verl/checkpoints/verl/Qwen2.5-7B-Instruct_qwen25_7b_17k_mathhard-7bhard/global_step_160/actor"
OUTPUT_DIR="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/models/Qwen2.5-7B-Instruct_qwen25_7b_17k_mathhard-7bhard"
WORLD_SIZE=8

# Parallel worker count (recommended: number of CPU cores)
NUM_WORKERS=$(nproc)  # Auto-detect CPU core count
# Or set manually: NUM_WORKERS=16

echo "=================================================="
echo "🚀 VERL Fast Conversion Mode"
echo "=================================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "World Size: $WORLD_SIZE"
echo "Parallel Workers: $NUM_WORKERS"
echo "=================================================="

# Show system resources
echo "💻 System Resources:"
echo "  CPU cores: $(nproc)"
echo "  Memory:"
free -h | grep -E "Mem|Swap"
echo ""

# Environment tweaks
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run conversion
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
