import asyncio
import json
import os
import hashlib
from typing import List, Dict, Any, Generator, Set

# Local imports
from post_api import CustomAPI
from prompt_loader import load_prompts
import strategies  # Strategy functions live in strategies.py

# ==============================================================================
# --- Pipeline core logic ---
# ==============================================================================

# Feature 1: stream JSONL to avoid loading everything into memory.
def stream_jsonl(file_path: str, id_key: str = "id") -> Generator[Dict[str, Any], None, None]:
    """
    Stream-read a JSONL file line-by-line (generator).

    If an item is missing the given id_key, generate a deterministic hash ID from its content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    # If a unique ID is missing, derive one from content to make the pipeline resumable.
                    if id_key not in item:
                        # Use a sorted JSON string to ensure determinism.
                        deterministic_string = json.dumps(item, sort_keys=True)
                        item[id_key] = hashlib.sha256(deterministic_string.encode('utf-8')).hexdigest()
                    yield item
                except json.JSONDecodeError:
                    print(f"Warning: skipping an unparsable line: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: input file not found: '{file_path}'. Exiting.")
        exit()

# Feature 2: collect processed IDs for resuming.
def get_processed_ids(file_path: str, id_key: str = "id") -> Set[Any]:
    """
    Read processed IDs from an output JSONL file for resuming.
    """
    processed_ids = set()
    if not os.path.exists(file_path):
        return processed_ids
    
    print(f"Found existing output file '{file_path}'. Reading processed IDs...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if id_key in item:
                    processed_ids.add(item[id_key])
            except json.JSONDecodeError:
                # Ignore corrupted lines in the output file.
                continue
    print(f"Found {len(processed_ids)} processed items.")
    return processed_ids


# save_jsonl supports append mode.
def save_jsonl(data: List[Dict[str, Any]], file_path: str, mode: str = 'w'):
    """
    Save data to a JSONL file.

    mode 'w': overwrite
    mode 'a': append
    """
    if not data:
        # In append mode, "no data" is expected and does not need logging.
        if mode == 'w':
            print(f"No data to save to '{file_path}'.")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    action = "wrote" if mode == 'w' else "appended"
    print(f"Successfully {action} {len(data)} items to '{file_path}'.")


def _ensure_failure_meta(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure failed items contain basic failure metadata."""
    ensured = []
    for it in items:
        if isinstance(it, dict) and '_failure' not in it:
            enriched = it.copy()
            enriched['_failure'] = {"step": "unknown", "reason": "unspecified", "attempts": 0}
            ensured.append(enriched)
        else:
            ensured.append(it)
    return ensured


async def main(
    strategy: str,
    input_file: str,
    output_file: str,
    prompts_file: str,
    api_urls: List[str],
    api_type: str,
    api_token: str,
    max_concurrent_requests: int,
    timeout: int,
    # Feature: batch processing + configurable ID key.
    batch_size: int,
    id_key: str = "id",
    # Whether to reprocess historical failures (default False skips failed IDs).
    reprocess_failed: bool = False,
):
    """
    Main data construction pipeline entry.

    Supports resuming and batched processing.
    """
    print("--- Pipeline starting (resume supported) ---")
    print(f"Strategy: {strategy}")
    print(f"Input:    {input_file}")
    print(f"Output:   {output_file}")
    print(f"Batch size: {batch_size}")
    print("-----------------------------------------")

    # 1) Load prompt templates
    templates = load_prompts(prompts_file)

    # 2) Resume logic: collect processed IDs from success/failure files
    processed_ids = get_processed_ids(output_file, id_key)
    failed_base, failed_ext = os.path.splitext(output_file)
    failed_output_file = f"{failed_base}_failed{failed_ext}"
    # Also check the failure file to avoid reprocessing failed items (configurable).
    failed_ids = get_processed_ids(failed_output_file, id_key)
    if reprocess_failed:
        print("Reprocess-failed is enabled: failed IDs will NOT be skipped.")
    else:
        processed_ids.update(failed_ids)
    
    if processed_ids:
        print(f"Total: will skip {len(processed_ids)} already-processed items (success or failure).")
    # Print historical counts to explain why a run may finish "instantly".
    if os.path.exists(output_file):
        print(f"Historical successes: {len(get_processed_ids(output_file, id_key))}")
    if os.path.exists(failed_output_file):
        print(f"Historical failures: {len(failed_ids)}")

    # 3) Initialize API client
    api_client = CustomAPI(
        url=api_urls[0],
        api_urls=api_urls,
        sk_token=api_token,
        api_type=api_type,
        timeout=timeout
    )

    # 4) Resolve the strategy function
    strategy_func = getattr(strategies, strategy, None)
    if not strategy_func or not callable(strategy_func):
        print(f"Error: strategy '{strategy}' was not found in strategies.py or is not callable.")
        available_strategies = [s for s in dir(strategies) if callable(getattr(strategies, s)) and not s.startswith("__")]
        print(f"Available strategies: {available_strategies}")
        return

    # 5) Run the strategy in batches
    print(f"Running strategy '{strategy}' (max_concurrent={max_concurrent_requests})...")
    
    original_infer_batch = api_client.infer_batch_async
    async def configured_infer_batch(*args, **kwargs):
        if 'max_concurrent' not in kwargs:
            kwargs['max_concurrent'] = max_concurrent_requests
        return await original_infer_batch(*args, **kwargs)
    api_client.infer_batch_async = configured_infer_batch
    
    batch_to_process = []
    total_processed_in_run = 0
    
    # Stream input and process in batches
    for item in stream_jsonl(input_file, id_key):
        # Skip processed items
        if item.get(id_key) in processed_ids:
            continue
        
        batch_to_process.append(item)
        
        # Process a full batch
        if len(batch_to_process) >= batch_size:
            print(f"\n--- Processing a batch of {len(batch_to_process)} items ---")
            completed_data, failed_data = await strategy_func(
                api_client=api_client,
                data=batch_to_process,
                templates=templates
            )
            
            # 6) Append-write results
            save_jsonl(completed_data, output_file, mode='a')
            if failed_data:
                save_jsonl(_ensure_failure_meta(failed_data), failed_output_file, mode='a')

            total_processed_in_run += len(batch_to_process)
            print(f"--- Batch finished. Total new items processed in this run: {total_processed_in_run} ---")
            
            # Reset batch
            batch_to_process = []

    # Process the final partial batch
    if batch_to_process:
        print(f"\n--- Processing the final batch of {len(batch_to_process)} items ---")
        completed_data, failed_data = await strategy_func(
            api_client=api_client,
            data=batch_to_process,
            templates=templates
        )

        save_jsonl(completed_data, output_file, mode='a')
        if failed_data:
            save_jsonl(_ensure_failure_meta(failed_data), failed_output_file, mode='a')

        total_processed_in_run += len(batch_to_process)
        print("--- Final batch finished. ---")

    print("\n--- Pipeline finished. ---")
    print(f"New items processed in this run: {total_processed_in_run}")
    final_success_count = len(get_processed_ids(output_file, id_key))
    final_failed_count = len(get_processed_ids(failed_output_file, id_key))
    print(f"Output file '{output_file}' now contains {final_success_count} items.")
    print(f"Failure file '{failed_output_file}' now contains {final_failed_count} items.")


if __name__ == "__main__":
    # Available strategies (each returns: completed_items, failed_items):
    # - "generate_degraded_question_and_info":
    #     Generate degraded_question, degraded_info, and required_points (missing-info checklist).
    #     Often used as a prerequisite step for other strategies.
    # - "generate_overconfidence_question_and_info":
    #     Generate overconfidence_question, overconfidence_info, and misleading_points.
    # - "generate_multi_turn_degraded_training_data":
    #     Full AskMind multi-turn generation: ask -> simulate user -> coverage check -> final answer -> judge -> force-correct (if needed).
    # - "generate_multi_turn_overconfidence_training_data":
    #     Overconfidence multi-turn generation: correct misleading claims, then answer and judge.
    # - "strategy_direct_answer_and_correct":
    #     Direct answer then judge; if wrong, reconstruct a "perfect answer" from expected_answer and optional solution.
    STRATEGY = os.getenv("STRATEGY", "generate_multi_turn_degraded_training_data")
    INPUT_FILE = os.getenv("INPUT_FILE", "/path/to/input.jsonl")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "/path/to/output.jsonl")
    API_URLS = [
        u.strip()
        for u in os.getenv("API_URLS", "http://127.0.0.1:8000/v1/chat/completions").split(",")
        if u.strip()
    ]
    API_TYPE = os.getenv("API_TYPE", "default")
    API_TOKEN = os.getenv("API_TOKEN", "none")
    DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.txt")
    PROMPTS_FILE = os.getenv("PROMPTS_FILE", DEFAULT_PROMPTS_FILE)
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "200"))
    TIMEOUT = int(os.getenv("TIMEOUT", "3600"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
    ID_KEY = os.getenv("ID_KEY", "id")
    # Enable via env var REPROCESS_FAILED=1 (takes precedence over the constant below).
    REPROCESS_FAILED = False  # Set to True to reprocess historical failures
    env_flag = os.getenv("REPROCESS_FAILED")
    if env_flag is not None:
        REPROCESS_FAILED = env_flag.strip() in ("1", "true", "True", "YES", "yes")
    
    asyncio.run(main(
        strategy=STRATEGY,
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        prompts_file=PROMPTS_FILE,
        api_urls=API_URLS,
        api_type=API_TYPE,
        api_token=API_TOKEN,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
        timeout=TIMEOUT,
        # Pass newly added configs.
        batch_size=BATCH_SIZE,
        id_key=ID_KEY,
        reprocess_failed=REPROCESS_FAILED,
    ))
