## Reward modules (for LLM-assisted debugging/modification)

This folder contains two **VERL-compatible** reward modules that implement the paper’s rubric-guided, turn-level shaping:

- `reward/ask_mind_qa.py`: AskMind (intent-deficient / missing-info) reward
- `reward/overconfidence_qa.py`: AskOverconfidence (misleading-premise / unjustified certainty) reward

Both files expose a top-level entry function that matches VERL’s reward interface:

- AskMind: `compute_score_ask_mind_qa(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float`
- Overconfidence: `compute_score_overconfidence_qa(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float`

### High-level control flow (both modules)

1) **`compute_score_*` parses `extra_info`**
   - decides whether the current step is a final turn (`is_final_turn`)
   - collects the original question, current question, conversation context, and rubric items

2) **Non-final turn shaping**
   - AskMind uses a checklist `required_points: List[str]`
   - Overconfidence uses a checklist `misleading_points: List[str]`

   Both prompt the judge to output a strict JSON object:
   - `answered_final: bool` (premature final answer at a non-final step)
   - `hits: List[bool]` (per-rubric-item coverage; list length must match the checklist)
   - `irrelevant_or_redundant: bool`
   - `notes: List[str]` (optional short notes; not used for scoring)

   The code maps judge outputs into the discrete reward set from the paper:

   - `-2.0` if `answered_final == True`
   - `-0.8` if `answered_final == False` and coverage is 0
   - `0.8` if partial coverage
   - `1.0` if full coverage

3) **Final turn grading**
   - judge returns `{"decision": "still_asking" | "wrong" | "correct"}`
   - mapped to `-2.0 / -1.0 / 1.0`

4) **Networking + robustness**
   - `call_llm_api_json()` calls an OpenAI-compatible `/v1/chat/completions` endpoint
   - strict JSON-only system prompt + tolerant parsing (`_extract_json`)
   - random endpoint selection over `API_URLS`, with retries
   - all failures fall back to conservative defaults (`DEFAULT_*_FAIL`)

### Key parameters to edit

- `API_URLS`: list of judge endpoints (OpenAI-compatible)
- `JUDGE_MODEL_NAME`: must match vLLM `--served-model-name`
- reward shaping constants: `DEFAULT_NON_FINAL_FAIL`, `DEFAULT_FINAL_FAIL`
- JSON schema / prompts: keep the schema stable if you want predictable parsing

### Expected `extra_info` schemas (minimal)

AskMind (`ask_mind_qa.py`):

- `is_final_turn: bool`
- `ori_question: str`
- `degraded_info: str`
- `required_points: List[str]` (preferred; enables checklist mode)
- `question: str` (current user turn)
- `context: str` (conversation history)
- `expected_answer: str` (final-turn target; falls back to `ground_truth`)

AskOverconfidence (`overconfidence_qa.py`):

- `is_final_turn: bool`
- `ori_question: str`
- `overconfidence_info: str`
- `misleading_points: List[str]` (preferred; enables checklist mode)
- `question: str`
- `context: str`
- `expected_answer: str` (final-turn target; falls back to `ground_truth`)
