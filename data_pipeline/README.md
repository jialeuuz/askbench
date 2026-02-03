# data_pipeline

Build AskBench-style multi-turn dialogue data (AskMind / AskOverconfidence) from single-turn QA examples, with an optional short-path strategy (“direct answer → judge → correction”).

Besides training trajectories, the same pipeline can also be used to convert other QA benchmarks into **AskMind/AskOverconfidence-style evaluation data** (by generating variant questions + checklist/rubrics), making it easy to extend AskBench to new domains.

- **AskMind (missing information / intent-deficient)**: degrade the original question into `degraded_question`, generate a checklist `required_points`, then run a multi-turn loop (ask → simulated user reply → answer → judge). If needed, force-correct the final answer to ensure correctness.
- **AskOverconfidence (false premises / misleading claims)**: keep the original givens verbatim while injecting confidently-stated wrong intermediate claims, producing `overconfidence_question` and `misleading_points`. The assistant must identify and correct the misleading points before answering.

For implementation details (code structure, prompt variable injection, strategy internals), see `readme_for_ai.md`. A Chinese copy of the original documentation is preserved as `data_pipeline/README_zh.md`.

## Setup

- Python 3.9+
- Install dependencies (either option works):
  - From repo root: `pip install -r data_pipeline/requirements.txt`
  - Or inside the directory: `cd data_pipeline && pip install -r requirements.txt`
- You need an OpenAI-compatible Chat Completions API that returns `choices[0].message.content`

## I/O Format

Input: JSONL (one example per line), with at least:

```json
{
  "id": "optional; deterministically hashed from content if missing",
  "ori_question": "original question",
  "expected_answer": "reference answer",
  "solution": "optional; used as a reference when force-correcting"
}
```

Output: JSONL (successful examples). Common fields:

- `conversation_history`: list of multi-turn messages, each like `{ "role": "user|assistant", "content": "..." }`
- `degraded_question` / `degraded_info` / `required_points`: AskMind fields
- `overconfidence_question` / `overconfidence_info` / `misleading_points`: AskOverconfidence fields

Failures: a JSONL file with the same name plus a `_failed` suffix. Each item includes `_failure` metadata (failed step, reason, retry count, truncated response preview, etc.).

## Running

Simplest: edit the constants at the bottom of `data_pipeline/main.py`, then run:

```bash
python data_pipeline/main.py
```

Or call `main()` from your own script (note: it is `async`):

```python
import asyncio
import sys

# Make Python able to import script-style modules under data_pipeline/ (main.py / strategies.py / post_api.py ...).
sys.path.append("data_pipeline")
from main import main

asyncio.run(main(
    strategy="generate_multi_turn_degraded_training_data",
    input_file="/path/to/input.jsonl",
    output_file="/path/to/output.jsonl",
    prompts_file="data_pipeline/prompts.txt",
    api_urls=["http://host:port/v1/chat/completions"],
    api_type="default",
    api_token="none",
    max_concurrent_requests=200,
    timeout=3600,
    batch_size=1000,
    id_key="id",
    reprocess_failed=False,
))
```

Resuming: by default, the pipeline skips `id`s that already exist in the success/failure files. To reprocess historical failures, set `REPROCESS_FAILED=1`.

## Strategies

Implemented in `data_pipeline/strategies.py`. Built-in strategies:

1) `generate_degraded_question_and_info`: generate `degraded_question` / `degraded_info` / `required_points`
2) `generate_overconfidence_question_and_info`: generate `overconfidence_question` / `overconfidence_info` / `misleading_points`
3) `generate_multi_turn_degraded_training_data`: AskMind multi-turn generation (ask → simulate user → coverage check → answer → judge → force-correct if needed)
4) `generate_multi_turn_overconfidence_training_data`: AskOverconfidence multi-turn generation (correct misleading points, then answer)
5) `strategy_direct_answer_and_correct`: short path (direct answer → judge; if wrong, reconstruct a “perfect answer” from `expected_answer` and optional `solution`)

## Parallel scheduling (optional)

`data_pipeline/run_queue.sh` + `data_pipeline/main_queue.py` support process-level parallel scheduling of multiple task queues (useful when you have multiple API endpoints or multiple input shards).
