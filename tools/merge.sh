#!/bin/bash
# merge.sh
set -e

# Override via env vars if desired (e.g., CHECKPOINT_DIR=... OUTPUT_DIR=... bash tools/merge.sh)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-verl/checkpoints/verl/Qwen2.5-7B-Instruct_test/global_step_160/actor}"
OUTPUT_DIR="${OUTPUT_DIR:-train/models/Qwen2.5-7B-Instruct_test}"
WORLD_SIZE="${WORLD_SIZE:-8}"

# Path to the VERL checkpoint conversion script in your environment.
# This repo does not vendor merge_verl.py; update this to match your setup.
MERGE_SCRIPT_PATH="${MERGE_SCRIPT_PATH:-/path/to/merge_verl.py}"

# Parallel worker count (recommended: number of CPU cores)
if command -v nproc >/dev/null 2>&1; then
    DEFAULT_NUM_WORKERS="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    DEFAULT_NUM_WORKERS="$(sysctl -n hw.ncpu 2>/dev/null || echo 16)"
else
    DEFAULT_NUM_WORKERS=16
fi

NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"
# Or set manually: NUM_WORKERS=16

echo "=================================================="
echo "🚀 VERL Fast Conversion Mode"
echo "=================================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "World Size: $WORLD_SIZE"
echo "Parallel Workers: $NUM_WORKERS"
echo "Merge script: $MERGE_SCRIPT_PATH"
echo "=================================================="

# Show system resources
echo "💻 System Resources:"
if command -v nproc >/dev/null 2>&1; then
    echo "  CPU cores: $(nproc)"
fi
if command -v free >/dev/null 2>&1; then
    echo "  Memory:"
    free -h | grep -E "Mem|Swap" || true
fi
echo ""

# Environment tweaks
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Basic validation
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ CHECKPOINT_DIR not found: $CHECKPOINT_DIR"
    exit 1
fi
if [ ! -f "$MERGE_SCRIPT_PATH" ]; then
    echo "❌ MERGE_SCRIPT_PATH not found: $MERGE_SCRIPT_PATH"
    echo "   Please set MERGE_SCRIPT_PATH to your local merge_verl.py."
    exit 1
fi

# Run conversion
if time python -u "$MERGE_SCRIPT_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --world_size "$WORLD_SIZE" \
    --num_workers "$NUM_WORKERS"; then
    echo ""
    echo "=================================================="
    echo "✅ SUCCESS!"
    echo "=================================================="
    ls -lh "$OUTPUT_DIR"
else
    echo "❌ Failed!"
    exit 1
fi
