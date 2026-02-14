# askQ: LLM-oriented repository index

This file is a **navigation hub** for LLM-assisted debugging/modification. It summarizes what each module does and where the key entry points are.

For code-level details, open each module’s own `readme_for_ai.md` (linked below).

Chinese version: `readme_for_ai_zh.md`.

## Quick routing (start here)

| If the user task is about... | Start with | Then read (LLM notes) |
| --- | --- | --- |
| Running evaluations (single-turn + AskBench multi-turn) | `ask_eval/README.md` | `ask_eval/readme_for_ai.md` |
| Building AskBench-style training dialogues offline | `data_pipeline/README.md` | `data_pipeline/readme_for_ai.md` |
| RLVR rewards / VERL integration | `reward/readme` | `reward/readme_for_ai.md` |
| Serving a local model as OpenAI-compatible API (vLLM) | `tools/vllm.sh` | `README.md` (Tools section) |
| Converting training checkpoints for vLLM | `tools/merge.sh` | `README.md` (Tools section) |
| Paper terminology / background | [arXiv:2602.11199](https://arxiv.org/abs/2602.11199) | `README.md` |

## Concepts (paper alignment)

- **AskBench**: interactive benchmark that turns standard QA into a judge-driven multi-turn protocol (candidate asks → judge simulates user → candidate answers → judge grades).
- **AskMind**: intent-deficient / missing-info queries; models should ask targeted clarification questions using a checklist (`required_points`).
- **AskOverconfidence**: misleading-premise queries; models should identify/correct false claims and avoid unjustified certainty using a checklist (`misleading_points`).
- **Rubric-guided RLVR**: turn-level shaping that rewards checklist coverage and penalizes premature final answers; plus final-turn correctness reward.

## Common end-to-end workflows (high level)

### A) Evaluate a local model through an API

1) (Optional) Convert a training checkpoint into an inference-ready HF directory: `tools/merge.sh`
2) Serve the model as an OpenAI-compatible API: `tools/vllm.sh`
3) Point `ask_eval/config/base.ini` (or `ask_eval/run.sh`) to the API URL
4) Run evaluation: `cd ask_eval && ./run.sh` (or `python scripts/main.py --config config/base.ini`)

### B) Build AskBench-style training dialogues (offline)

1) Prepare an input JSONL with at least `ori_question` + `expected_answer` (and optionally `solution`)
2) Choose a strategy in `data_pipeline/strategies.py`
3) Run `data_pipeline/main.py` to generate multi-turn trajectories, plus a `_failed.jsonl` for retries/debugging

### C) Train with VERL + RLVR rewards

1) Copy reward modules into VERL and register them in `default_compute_score()` (see `reward/readme`)
2) Configure the judge model API endpoints in the reward modules (`API_URLS` / `JUDGE_MODEL_NAME`)
3) Use `reward/train.sh` as a **sanitized reference** launcher (paths and data are placeholders)

## Module map (what to open / what to edit)

### `ask_eval/` (evaluation)

- Purpose: runs both single-turn benchmarks and AskBench-style multi-turn judge loops.
- Main entry points:
  - `ask_eval/run.sh` (recommended runner; edits `config/base.ini` via overrides)
  - `ask_eval/scripts/main.py` (reads INI config and dispatches per task)
- Key config:
  - `ask_eval/config/base.ini` (candidate + judge API config; tasks; concurrency; max turns)
  - `ask_eval/config/common/*.ini` (per-task overrides)
- Outputs:
  - `ask_eval/results/...` (task logs + summaries)
- LLM notes: `ask_eval/readme_for_ai.md`

### `data_pipeline/` (data construction)

- Purpose: converts raw QA into AskBench-style multi-turn dialogues (or related variants) for training.
- Main entry points:
  - `data_pipeline/main.py` (single-job runner)
  - `data_pipeline/main_queue.py` + `data_pipeline/run_queue.sh` (optional multi-job scheduling)
- Key config/knobs:
  - strategy selection (see `data_pipeline/strategies.py`)
  - prompt templates: `data_pipeline/prompts.txt`
  - API client: `data_pipeline/post_api.py`
- Outputs:
  - success JSONL + `_failed.jsonl` (resume + debugging)
- LLM notes: `data_pipeline/readme_for_ai.md`

### `reward/` (rubric-guided RLVR rewards)

- Purpose: VERL-compatible reward functions for the paper’s two dimensions + a reference training script.
- Key files:
  - `reward/ask_mind_qa.py` (entry: `compute_score_ask_mind_qa`)
  - `reward/overconfidence_qa.py` (entry: `compute_score_overconfidence_qa`)
  - `reward/train.sh` (sanitized reference launcher; paths are placeholders)
- Integration docs: `reward/readme`
- LLM notes: `reward/readme_for_ai.md`

### `tools/` (ops: convert + serve)

- `tools/merge.sh`: converts/merges sharded training checkpoints into an inference-ready HuggingFace model directory.
- `tools/vllm.sh`: starts vLLM’s OpenAI-compatible server for serving local models.

## Naming conventions

- `README.md`: user-facing “how to run”.
- `readme_for_ai.md`: implementation-oriented notes for LLM debugging/modification.
- `*_zh.*`: Chinese versions of documentation.
