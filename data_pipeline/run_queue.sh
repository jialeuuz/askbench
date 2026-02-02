#!/bin/bash

# ==============================================================================
# --- Task configuration center (readability-first) ---
# ==============================================================================

# Usage:
# 1) Define one variable per *serial task queue* (e.g., SERIAL_QUEUE_1, SERIAL_QUEUE_2).
#    - Each variable is a JSON array string: `[{...}, {...}]`.
#    - Tasks inside a queue run sequentially.

# 2) Combine queues you want to run *in parallel* in the `PARALLEL_QUEUES` array.
#    - Each queue runs in its own parallel process.

# --- Serial queue 1 ---
SERIAL_QUEUE_1='[
  {
    "STRATEGY": "generate_multi_turn_degraded_training_data",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/dapo/dapo_distill_sample_2k.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_math_sample_20k_oss120b_heigh.jsonl",
    "API_URLS": ["http://10.80.13.242:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "generate_multi_turn_degraded_training_data",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/med_sample_20k_clear.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_med_sample_20k_oss120b_heigh.jsonl",
    "API_URLS": ["http://10.80.13.242:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "strategy_direct_answer_and_correct",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/med_sample_20k_clear.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/ori_med_sample_20k_oss120b_heigh.jsonl",
    "API_URLS": ["http://10.80.13.242:8012/v1/chat/completions"]
  }
]'

# --- Serial queue 2 (uses a different API; can run in parallel) ---
SERIAL_QUEUE_2='[
  {
    "STRATEGY": "generate_multi_turn_degraded_training_data",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/useless/math_sample_20k.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_math_sample_20k_a3b_ins_2507.jsonl",
    "API_URLS": ["http://10.80.12.34:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "strategy_direct_answer_and_correct",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/useless/math_sample_20k.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/ori_math_sample_20k_a3b_ins_2507.jsonl",
    "API_URLS": ["http://10.80.12.34:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "generate_multi_turn_degraded_training_data",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/med_sample_20k_clear.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_med_sample_20k_a3b_ins_2507.jsonl",
    "API_URLS": ["http://10.80.12.34:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "strategy_direct_answer_and_correct",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/med_sample_20k_clear.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/ori_med_sample_20k_a3b_ins_2507.jsonl",
    "API_URLS": ["http://10.80.12.34:8012/v1/chat/completions"]
  }
]'


# --- Parallel execution plan ---
# Add/remove queue variables below.
# Example: to run only the first queue: PARALLEL_QUEUES=( "$SERIAL_QUEUE_1" )
# The config below runs the queues defined above in parallel.
PARALLEL_QUEUES=(
  "$SERIAL_QUEUE_1"
  "$SERIAL_QUEUE_2"
)


# ==============================================================================
# --- Execution script (no changes needed) ---
# ==============================================================================
# This section combines the queues into the final JSON format expected by the Python scheduler.

# Check whether any queues are configured
if [ ${#PARALLEL_QUEUES[@]} -eq 0 ]; then
  echo "Warning: PARALLEL_QUEUES is empty; no tasks to run."
  exit 0
fi

# Join all queues with commas and wrap with brackets to form a valid JSON array
TASK_CONFIG=$(IFS=,; echo "[${PARALLEL_QUEUES[*]}]")

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Task config loaded. Launching Python scheduler..."
echo "Working directory: $DIR"
echo "Parallel queues: ${#PARALLEL_QUEUES[@]}"
echo "--------------------------------------------------"

# Run the scheduler and pass the task config as a CLI argument.
# (cd ... && python3 ...) ensures the correct working directory.
(cd "$DIR" && python3 main_queue.py "$TASK_CONFIG")

echo "--------------------------------------------------"
echo "Shell script finished."
