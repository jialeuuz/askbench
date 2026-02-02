# data_pipeline/main_queue.py

import sys
import json
import multiprocessing
import asyncio
from datetime import datetime

# Assumes the main pipeline entry is `main.py`.
# Import `main()` and rename it to avoid confusion with this module.
try:
    from main import main as run_pipeline_task
except ImportError:
    print("Error: failed to import 'main' from 'main.py'.")
    print("Make sure 'main_queue.py' and 'main.py' are in the same directory.")
    sys.exit(1)

# ==============================================================================
# Global fixed defaults (shared across tasks)
# ==============================================================================
# These defaults can be overridden by per-task JSON keys of the same name.
FIXED_PARAMS = {
    "API_TYPE": "default",
    "API_TOKEN": "none",
    "PROMPTS_FILE": "prompts.txt",
    "MAX_CONCURRENT_REQUESTS": 200,
    "TIMEOUT": 600
}

def get_current_time():
    """Return current time as a formatted string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def worker_process(task_queue: list, worker_id: int):
    """
    Worker function executed by each parallel process.

    Tasks in a worker are executed sequentially, in order.
    
    Args:
        task_queue: list of task dicts, e.g. [{'STRATEGY': 's1', ...}, {'STRATEGY': 's2', ...}]
        worker_id: worker index used for logging
    """
    print(f"[{get_current_time()}] [worker {worker_id}] started with {len(task_queue)} sequential task(s).")
    
    for i, task_config in enumerate(task_queue):
        task_num = i + 1
        print("-" * 80)
        print(f"[{get_current_time()}] [worker {worker_id}] running task {task_num}/{len(task_queue)}...")
        
        # Merge fixed defaults with task-specific overrides.
        params = FIXED_PARAMS.copy()
        params.update(task_config)

        # Validate required keys
        required_keys = ['STRATEGY', 'INPUT_FILE', 'OUTPUT_FILE', 'API_URLS']
        if not all(key in params for key in required_keys):
            print(f"[{get_current_time()}] [worker {worker_id}] error: task {task_num} is missing required keys; skipping.")
            print(f"    Required keys: {required_keys}")
            print(f"    Task config: {task_config}")
            continue

        try:
            # run_pipeline_task is async; execute it via asyncio.run().
            asyncio.run(run_pipeline_task(
                strategy=params['STRATEGY'],
                input_file=params['INPUT_FILE'],
                output_file=params['OUTPUT_FILE'],
                prompts_file=params['PROMPTS_FILE'],
                api_urls=params['API_URLS'],
                api_type=params['API_TYPE'],
                api_token=params['API_TOKEN'],
                max_concurrent_requests=params['MAX_CONCURRENT_REQUESTS'],
                timeout=params['TIMEOUT']
            ))
            print(f"[{get_current_time()}] [worker {worker_id}] completed task {task_num}/{len(task_queue)}.")
        except Exception as e:
            print(f"[{get_current_time()}] [worker {worker_id}] error: task {task_num}/{len(task_queue)} failed.")
            print(f"    Details: {e}")
            # Continue to the next task even if one fails.
    
    print(f"[{get_current_time()}] [worker {worker_id}] all assigned tasks finished.")


if __name__ == "__main__":
    # Use 'spawn' for better portability/stability across platforms.
    multiprocessing.set_start_method("spawn", force=True)

    if len(sys.argv) != 2:
        print("Usage: python main_queue.py '<JSON_CONFIG_STRING>'")
        print("Example: python main_queue.py '[[{\"STRATEGY\": ...}]]'")
        sys.exit(1)

    json_config_string = sys.argv[1]
    
    try:
        parallel_queues = json.loads(json_config_string)
        if not isinstance(parallel_queues, list):
            raise ValueError("JSON config must be a list.")
    except (json.JSONDecodeError, ValueError) as e:
        print("Error: failed to parse JSON config string. Please check the format.")
        print(f"    Details: {e}")
        print(f"    Received: {json_config_string}")
        sys.exit(1)

    num_parallel_workers = len(parallel_queues)
    if num_parallel_workers == 0:
        print("Empty config: no tasks to run.")
        sys.exit(0)

    print("=" * 80)
    print(f"Scheduler starting. Launching {num_parallel_workers} parallel worker(s).")
    print("=" * 80)

    # Prepare args for each worker (task queue + worker ID)
    worker_args = [(queue, i + 1) for i, queue in enumerate(parallel_queues)]

    # Create and run the process pool.
    with multiprocessing.Pool(processes=num_parallel_workers) as pool:
        # starmap unpacks each tuple in worker_args as arguments to worker_process().
        pool.starmap(worker_process, worker_args)

    print("=" * 80)
    print(f"[{get_current_time()}] All parallel workers finished. Scheduler exiting.")
    print("=" * 80)
