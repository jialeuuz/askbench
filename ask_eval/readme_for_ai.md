## Quick map

This document is implementation-oriented (for maintainers / debugging / LLM-assisted edits). The original Chinese version is preserved as `ask_eval/readme_for_ai_zh.md`.

| Path | Purpose |
| --- | --- |
| `scripts/main.py` | Entry point: reads task config and dispatches the corresponding runner |
| `scripts/run.py` | Single-turn evaluation loop (Math / MedQA / etc.) |
| `scripts/run_ask.py` | AskBench-style multi-turn evaluation (candidate model + judge model) |
| `ask_eval/models/` | Model wrappers: unified API calls, batched inference, keep-alive logic |
| `ask_eval/data/` | Dataset loaders (currently defaults to `test.jsonl`) |
| `ask_eval/evaluators/` | Evaluators and scoring strategies |
| `ask_eval/utils/config.py` | Config parsing/merging and result aggregation utilities |
| `config/base.ini` | Global defaults; `config/common/*.ini` are per-task overrides |
| `run.sh` | Convenience script to override config values and launch evaluation |
| `data/fata/*` | FATA task data folders (one `test.jsonl` per task) |

## Execution flow

1) **Load base config**: `scripts/main.py` reads the INI file provided by `--config` and parses `[tasks] enabled`.
2) **Dispatch per task**:
   - By default, merge in `config/common/<task>.ini` and run `scripts/run.py`.
   - If the task name contains `fata`, `ask`, or `quest_bench`, run `scripts/run_ask.py` to execute the judge-driven multi-turn loop (FATA details below).
   - If `tasks_config_path` points to an alternative template family (e.g., EvalScope/OpenCompass), `main.py` switches to the corresponding branch and writes a unified final metric summary at the end.
3) **Write outputs**:
   - Single-turn: `api_responses.json`, `summary_results.json`, `results.txt`.
   - AskBench-style: `askbench_detailed_results.json`, `results.txt`.
   - After all tasks finish, `write_final_result_file` (or `write_final_evalscope_result_file`) appends a summary line.

In one line:

```
INI config -> merge per-task config -> load data -> batch inference
        -> evaluator compare / judge loop -> write per-attempt logs + summaries
```

## Config system

- **Base config (`config/base.ini`)**
  - `[model]`: candidate model API, auth, system prompt, etc.
  - `[generateconfig]`: generation params (`max_tokens`, `temperature`, `max_concurrent`, `n_attempts`, ...).
  - `[tasks]`: task switches and the per-task config directory.
  - `[path]`: default data roots and result roots.
  - `[evaluatorconfig]`: judge model config. This is used not only by AskBench tasks, but also reused by some single-turn tasks (math500 / medqa / gpqa / bbh).
- **Per-task config (`config/common/<task>.ini`)**
  - Overrides data paths, task aliases, default APIs, etc.
  - `load_merged_config` applies overrides section-by-section on top of the base config.
- **Runner convenience script (`run.sh`)**
  - Edit variables at the top to override config values (model URL, task list, temperature, concurrency, ...).
  - Backs up `base.ini` before the run and restores it afterwards to avoid polluting defaults.
  - `max_turns` defaults to 3 in `run.sh`. You can change `[evaluatorconfig] max_turns` in `config/base.ini`, or explicitly pass `./run.sh --max-turns N`.
  - `GUIDANCE_MODE` controls first-turn guidance: `none` (default), `weak`, `strong`, `fata`. `fata` appends the official FATA guidance text to the first user message to compare against the baseline protocol.
  - Set `STRICT_MODE=1` to enable AskBench strict mode: force a two-turn protocol (turn 1 must clarify/correct; turn 2 must provide a final answer and must not clarify again). The judge is also stricter (the final answer must be unique; “contains a correct answer somewhere” is still marked wrong). With `STRICT_MODE=0`, the prompt and evaluation flow remain unchanged.

## Data loading

`ask_eval/data/data_map.py` maps task names to loaders (currently all `JsonlLoader`), defaulting to `data/<group>/<task>/test.jsonl`. Schemas vary slightly:

- **Math / general QA**: contains `problem` or `ori_question` plus `expected_answer`.
- **Degraded-question tasks**: use `degraded_question` plus `expected_answer`.
- **AskBench family**: contains `degraded_question`, `ori_question`, `expected_answer`, `degraded_info`, and `required_points` (a checklist of missing key information; currently supported for `ask_mind*` and `quest_bench`), used by the multi-turn judge loop.
- **`in3_interaction`**: upstream data provides `task`, `vague`, `missing_details`, and example interactions. The evaluator renames `task` into `ori_question`/`degraded_question`, converts each `missing_details[i].description` into `required_points`, and rewrites the full `missing_details` into `degraded_info`. Since there is no `expected_answer`, this benchmark only reports clarification behavior metrics (ask-rate / coverage / redundant questions, etc.), not Accuracy.
- **HealthBench**: `prompt` directly contains a multi-turn message list (`role`/`content`), and `rubrics` is a list of scoring items with points. The candidate model generates a reply on top of the existing conversation, then the full dialogue plus a single rubric item is sent to the judge (template: `data/common/healthbench/grader_prompt.py`). The judge returns a (possibly negative) score for that rubric. The final score is a normalized 0–1 value from the sum of positive points (clipped at 0).
- **AskMind combined set**: `data/ask_bench/ask_mind/test.jsonl` is a concatenation of 100 examples from each of `ask_mind_math500de/medqade/gpqade/bbhde` (400 total). You can reproduce it with `python data/ask_bench/ask_mind/build_combined_eval.py`. The `ask_mind` task uses the same AskBench logic and the same composite score.

To add a new task, register a custom loader in `LOADER_MAP`, or keep the same JSONL convention.

## Model abstraction

All models inherit from `ask_eval/models/base_api_model.BaseAPIModel`, which provides unified sync/async inference APIs.

- **`CustomAPI`**: for self-hosted inference services. Supports `enable_thinking`, custom headers, and optional splitting of `<think></think>` blocks.
- **`GPTAPI`**: for GPT-4o / GPT-5 style APIs, with built-in QPS control and request formatting for images and developer prompts.
- **Health checks**: `create_model` calls `check_urls_health` after instantiation to ensure at least one URL is reachable.

Batch inference uses `infer_batch_async` with `max_concurrent` to cap concurrency, and returns the response text, extracted “thinking” (if enabled), and truncation flags.

## Evaluators

Evaluators inherit from `ask_eval/evaluators/base_evaluator.BaseEvaluator` and implement:

- `format_example`: build the prompt to send to the model (supports few-shot).
- `extract_answer`: extract a candidate answer from the model output.
- `validate_answer`: compare prediction vs. reference (sync/async).
- `evaluate_responses`: write `api_responses.json` and compute aggregate metrics (accuracy, truncation rate, ...).

Task-to-evaluator wiring is defined in `ask_eval/evaluators/evaluator_map.EVALUATOR_MAP`. Common patterns:

- **Math**: `MathEvaluator` supports LaTeX normalization, SymPy simplification, numeric matching, etc. `Math500Evaluator` and others override prompts/extraction as needed.
- **MedQA / GPQA / etc.**: domain-specific extraction and validation logic.
- **`AskEvaluator`**: the multi-turn core. The judge model plays three roles: classify clarification vs. final answer, evaluate final-answer correctness, and simulate user follow-ups when the assistant asks questions. The loop is:
  1) Candidate model produces the next reply (the last turn is forced to be a final answer).
  2) Judge decides whether the reply is a final answer and whether it is correct.
  3) If not final and turns remain, the judge generates a human-like user follow-up based on the hidden `ori_question` and scenario context (e.g., `degraded_info` / `overconfidence_info`).
  4) Append the turn to the dialogue trace until the model answers or the budget is exhausted.

### Single-turn judge grading

`math500`, `medqa`, `gpqa`, and `bbh` are graded by the judge model to avoid brittle regex-only matching:

- For each example, the prompt, the reference answer, and the regex-extracted candidate answer are all provided to the judge.
- The judge must output a `Reasoning: ...` line first, then a JSON code block with a fixed schema: `{"reason": "...", "result": "correct" | "incorrect"}`.
- JSON parse failures are automatically retried up to 10 times. If it still fails, the example is skipped (`skipped=true`), excluded from accuracy/Pass@1, and the failure reason is recorded in `api_responses.json`.
- The judge config in `[evaluatorconfig]` is reused across these tasks.

`askbench_detailed_results.json` additionally records full dialogue traces (when applicable), judge decisions, and failure statistics.

## Outputs

Standard single-turn tasks produce:

- `api_responses.json`: per-example details (raw output, extracted answer, correctness, chain-of-thought if available, truncation flags).
- `summary_results.json`: per-example aggregation across multiple attempts (including `pass@1`).
- `results.txt`: human-readable summary (accuracy / Pass@1, runtime, truncation summary, ...).
- `results/final_result.txt`: an appended one-line summary after all tasks finish.

AskBench-style tasks additionally write `askbench_detailed_results.json` (turn-by-turn logs + failure reason distribution). For EvalScope / OpenCompass-style paths, the framework finds the newest timestamped folder under the task directory and parses `results.txt` to keep a unified final summary format.

## AskMind / AskOverconfidence metric extensions

- **New data field**: `data/ask_bench/ask_mind/*/test.jsonl` and `data/ask_bench/quest_bench/test.jsonl` include `required_points`, listing every missing/blurred key detail so the judge can determine whether the model asked for all necessary information. Example:
  ```json
  {
    "degraded_question": "...",
    "required_points": [
      "Exact value of the first lifetime (10^-9 sec)",
      "Exact value of the second lifetime (10^-8 sec)"
    ]
  }
  ```
- **AskOverconfidence fields**: `data/ask_bench/ask_overconfidence/*/test.jsonl` uses `overconfidence_question`, `overconfidence_info`, and a checklist `misleading_points` (with `required_points` supported as an alias). These correspond to: the user-facing question with injected misleading claims, a description of wrong assertions vs. correct facts, and the set of misleading points that must be challenged/corrected. `AskEvaluator` maps these fields into scenario context and checklist points. The field names can be unified, but the semantics still follow the overconfidence rules (the assistant must proactively identify and correct misleading claims).
  - **User simulation for overconfidence**: overconfidence uses a `simulator_model` to generate “accept/reject” style user replies. To reduce leakage risk, the simulator prompt does not include `ori_question/expected_answer`; it can only respond based on the misleading-points checklist.
  - **Decoupling simulator vs judge**: you can introduce `[simulatorconfig]` to use a different model/API from `[evaluatorconfig]` (Judge).
- **Combined evaluation set**: `ask_overconfidence` reads `data/ask_bench/ask_overconfidence/test.jsonl`, which is a mixture of 100 examples sampled from each subset (math500/medqa/gpqa/bbh). Rebuild with `data/ask_bench/ask_overconfidence/build_combined_eval.py` (also writes `source_task` for provenance).
- **Judge output contract**: `AskEvaluator` requires the judge to output a `Reasoning:` line, then a strict ```json block containing `is_final_answer`, `is_correct`, `all_required_points_resolved`, `missing_required_points`, and optional `notes`. JSON parse failures are retried up to 10 times; persistent failures mark the example as skipped (excluded from final metrics) with `JudgeJSONParseFailed`.
- **Metric attribution**: `askbench_detailed_results.json` records which `required_points` were covered on each turn, whether the assistant asked after all points were already resolved (redundant questioning), and whether the final answer was produced before resolving required points.
- **Reported metrics**: `results.txt` (and the CLI summary) reports:
  - accuracy over valid (non-skipped) examples;
  - `ask_rate`: fraction of valid examples where the model asked at least one clarification question (e.g., 300 out of 500 → 0.6);
  - `cov_rate`: whether all `required_points` were resolved before the final answer;
  - `unq_rate`: how often the assistant asked after the information was already complete (counted as examples and as events);
  - a composite `score`: used for `ask_mind_math500de/medqade/gpqade/bbhde`, `ask_overconfidence(+_math500/+_medqa/+_gpqa/+_bbh)`, and `quest_bench`, computed as `0.5 * acc + 0.3 * cov_rate + 0.2 * (1 - unq_rate)` (lower `unq_rate` is better);
  - full reason distributions (including skipped examples) for debugging.
- **Special case: `in3_interaction`**: uses the same “ask” behavior metrics, but without `expected_answer` the final logs only include behavioral metrics such as `Vague Ask Rate / Clear-task Direct Rate / cov_rate / unq_rate`. No Accuracy or composite score is reported. The first line of `results.txt` additionally records `Vague Ask Rate` so `final_result.txt` can aggregate it.

## FATA: two-stage protocol

- **Data source**: `fata_math500` and `fata_medqa` reuse AskMind data (copied from `ask_mind_math500de` and `ask_mind_medqade` respectively), stored under `data/fata/<task>/test.jsonl`.
- **Interaction**:
  1) The candidate model receives the official FATA prompt and sees the degraded question:
     ```
     User request: <degraded_question>.
     To better assist me, before offering advice, please adopt the perspective of an expert in the relevant field
     and ask questions to help you identify any missing key information.
     Please ensure the problem is structured clearly and expressed concisely, with example guidance,
     just like how experts ask users questions during consultations to gather key information before providing solutions.

     After I provide additional information, please then offer a more personalized and practical solution as an expert in that field.
     If all key information has already been provided, please directly give the solution.
     Note: Maintain a positive attitude, and do not request phone numbers, ID numbers, or other sensitive data.
     ```
     The model may ask one clarification question in the first turn, or answer directly.
  2) Every assistant reply is passed to the judge. Given `ori_question`, `degraded_info`, `required_points`, and `expected_answer`, the judge decides whether the assistant is asking for missing information:
     - If the assistant is indeed asking, the judge produces a user follow-up consistent with the original question and passes it to the model as turn 2.
     - If the assistant starts answering, the judge directly grades correctness.
  3) The official protocol is typically two turns (`max_turns=2`). In this framework, you can adjust turns via `[evaluatorconfig] max_turns` (or `./run.sh --max-turns N`). Under the two-turn setting, asking again on turn 2 violates the “ask once then answer” rule and is marked wrong.
- **Scoring**:
  - The judge outputs JSON with `needs_more_info`, optional `user_reply`, `is_correct`, and `reason`. When `needs_more_info=false`, correctness is judged against `expected_answer`.
  - Outputs follow AskBench conventions: `askbench_detailed_results.json` stores full dialogue traces and judge decisions, while `summary_results.json`/`results.txt` summarize accuracy, whether clarification was triggered, failure cases where the model kept asking on turn 2, and judge JSON-parse failures.

## Extending the framework

1) **New dataset**: create `data/<group>/<task>/test.jsonl`, implement or reuse a loader, and register it in `LOADER_MAP`.
2) **New evaluator**: inherit from `BaseEvaluator`, implement formatting/extraction/validation, and register the class in `EVALUATOR_MAP`.
3) **New task config**: add `config/common/<task>.ini` and set `[evalset] evalsetname`, model, generation, and path params.
4) **Multi-model sweeps**: edit variables at the top of `run.sh`, or wrap it in an outer loop script.

## Common parameters

- `generateconfig.n_attempts`: sample multiple times per question and report average accuracy and `pass@1`.
- `generateconfig.max_concurrent`: cap candidate-model concurrency to avoid overwhelming the service. AskBench/QuestBench multi-turn evaluation follows this cap strictly for async calls.
- `[evaluatorconfig].max_concurrent`: cap judge-model concurrency. The judge both arbitrates and simulates users, so it must also be throttled.
- `evaluatorconfig.max_turns`: max dialogue turns for AskBench-style tasks (default 5 in `base.ini`; `run.sh` overrides to 3 by default).
- `model.extra_prompt`, `model.system_prompt`: automatically composed into user/system messages via `BaseAPIModel.format_messages`.

## Notes / caveats

- If `tasks_config_path` contains `EvalScope`, `main.py` will attempt to call `scripts.run_evalscope_origin`. That script is not included in this repo; you must provide it before using that path.
- Default dependencies are listed in `setup.py` (e.g., `sympy`, `latex2sympy2` for math normalization).
- AskBench-style evaluation depends on judge stability/consistency. It is recommended to set judge temperature to 0 and cap concurrency.

## Built-in tasks (at a glance)

| Task id | Data dir | Evaluator | Protocol | Core behavior |
| --- | --- | --- | --- | --- |
| `math500` | `data/common/math500` | `Math500Evaluator` | single-turn + judge | Single-turn answering; judge grades correctness and logs skip/failure reasons. |
| `medqa` | `data/common/medqa` | `MedQAEvaluator` | single-turn + judge | Single-turn multiple-choice; judge grades using the question and reference JSON. |
| `gpqa` | `data/common/gpqa` | `GpqaEvaluator` | single-turn + judge | General QA; judge grades the model output. |
| `bbh` | `data/common/bbh` | `BBHEvaluator` | single-turn + judge | BBH full set; judge grades and supports both option-based and open-ended answers. |
| `ask_overconfidence` | `data/ask_bench/ask_overconfidence` | `AskEvaluator` | multi-turn + judge | AskBench overconfidence main task; default is a 400-example mixture (100 each from math500/medqa/gpqa/bbh; see `data/ask_bench/ask_overconfidence/test.jsonl`). |
| `ask_overconfidence_math500` | `data/ask_bench/ask_overconfidence` | `AskEvaluator` | multi-turn + judge | Overconfidence variant built from Math500; model must identify and correct misleading points before answering. |
| `ask_overconfidence_medqa` | `data/ask_bench/ask_overconfidence` | `AskEvaluator` | multi-turn + judge | Overconfidence variant built from MedQA (same schema as `ask_overconfidence_math500`). |
| `ask_overconfidence_gpqa` | `data/ask_bench/ask_overconfidence` | `AskEvaluator` | multi-turn + judge | Overconfidence variant built from GPQA. |
| `ask_overconfidence_bbh` | `data/ask_bench/ask_overconfidence` | `AskEvaluator` | multi-turn + judge | Overconfidence variant built from BBH. |
| `ask_mind` | `data/ask_bench/ask_mind` | `AskEvaluator` | multi-turn + judge | AskBench main task; assistant sees `degraded_question` while the hidden original is in `ori_question`. Default is a 400-example mixture (100 from each AskMind subset). |
| `ask_mind_math500de` | `data/ask_bench/ask_mind` | `AskEvaluator` | multi-turn + judge | AskMind + degraded Math500: tests proactive clarification for missing information. |
| `ask_mind_medqade` | `data/ask_bench/ask_mind` | `AskEvaluator` | multi-turn + judge | AskMind + degraded MedQA. |
| `ask_mind_gpqade` | `data/ask_bench/ask_mind` | `AskEvaluator` | multi-turn + judge | AskMind + degraded GPQA. |
| `ask_mind_bbhde` | `data/ask_bench/ask_mind` | `AskEvaluator` | multi-turn + judge | AskMind + degraded BBH. |
| `fata_math500` | `data/fata/fata_math500` | `AskEvaluator` | two-turn (clarify + answer) | Official FATA prompt encourages one clarification; judge simulates user follow-up if needed, then judges the final answer. |
| `fata_medqa` | `data/fata/fata_medqa` | `AskEvaluator` | two-turn (clarify + answer) | Same as `fata_math500`, but sourced from MedQA. |
| `quest_bench` | `data/ask_bench/quest_bench` | `AskEvaluator` | multi-turn + judge | QuestBench, evaluated with `AskEvaluator` and `required_points` (AskMind-style judge logic). |
| `in3_interaction` | `data/ask_bench/in3_interaction` | `In3InteractionEvaluator` | multi-turn + judge | IN3 Interaction: treats `task` as the original question, expands `missing_details` into `required_points`, and reports clarification behavior only (no Accuracy). |

> Note: all `ask_*`, `quest_bench`, and also `math500` / `medqa` / `gpqa` / `bbh` depend on the judge model defined in `[evaluatorconfig]`. Other “traditional” tasks may still rely on regex or numeric comparison. When adding a new task, this table is a quick way to locate the required schema and evaluator.
