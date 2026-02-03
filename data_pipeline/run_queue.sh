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
    "INPUT_FILE": "/path/to/input.jsonl",
    "OUTPUT_FILE": "/path/to/output.jsonl",
    "API_URLS": ["http://127.0.0.1:8000/v1/chat/completions"]
  }
]'

# --- Serial queue 2 (uses a different API; can run in parallel) ---
SERIAL_QUEUE_2='[
  {
    "STRATEGY": "generate_multi_turn_overconfidence_training_data",
    "INPUT_FILE": "/path/to/input_overconfidence.jsonl",
    "OUTPUT_FILE": "/path/to/output_overconfidence.jsonl",
    "API_URLS": ["http://127.0.0.1:8001/v1/chat/completions"]
  }
]'


# --- Parallel execution plan ---
# Add/remove queue variables below.
# Example: to run only the first queue: PARALLEL_QUEUES=( "$SERIAL_QUEUE_1" )
# Keep empty by default to avoid accidentally running placeholder tasks.
PARALLEL_QUEUES=()


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
