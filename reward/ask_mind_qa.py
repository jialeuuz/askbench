# verl/utils/reward_score/ask_mind_qa.py
import random
import time
from typing import List, Optional, Dict, Any
import requests
import json
import re

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Configure your judge model endpoints here (OpenAI-compatible /chat/completions).
# The default assumes a local vLLM OpenAI server (see tools/vllm.sh).
API_URLS = [
    "http://127.0.0.1:8012/v1/chat/completions",
    "http://127.0.0.1:8013/v1/chat/completions",
    "http://127.0.0.1:8014/v1/chat/completions",
    "http://127.0.0.1:8015/v1/chat/completions",
]
# Must match the server's `--served-model-name` (vLLM).
JUDGE_MODEL_NAME = "default"

# Conservative defaults on failure (do not return 0.0).
DEFAULT_NON_FINAL_FAIL = -0.8  # Non-final turn: parsing failures / exceptions.
DEFAULT_FINAL_FAIL = -1.0      # Final turn: parsing failures / exceptions.


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parsing:
    1) Prefer ```json ...``` fenced block.
    2) Otherwise try json.loads(text).
    3) Otherwise slice from the first '{' to the last '}'.
    """
    if not text:
        return None

    # 1) Fenced code block
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass

    # 2) Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3) First '{' ... last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Judge call (JSON-only)
# ---------------------------------------------------------------------------
def call_llm_api_json(prompt: str, max_retries: int = 10, max_tokens: int = 16000) -> Optional[Dict[str, Any]]:
    """
    Call the judge model and force JSON-only output.
    Retries across endpoints and uses tolerant JSON parsing.
    Returns None on failure; callers map it to conservative defaults.
    """
    for attempt in range(max_retries):
        try:
            api_urls = API_URLS
            if not api_urls:
                time.sleep(0.3)
                continue

            api_url = random.choice(api_urls)
            headers = {"Content-Type": "application/json"}
            data = {
                "model": JUDGE_MODEL_NAME,
                "messages": [
                    {"role": "system", "content":
                        "You are a strict JSON grader. Always return a single JSON object with no extra text, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens
            }

            resp = requests.post(api_url, headers=headers, json=data, timeout=3600)
            if resp.status_code == 200:
                result = resp.json()
                content = result["choices"][0]["message"]["content"]
                parsed = _extract_json(content)
                if parsed is not None:
                    return parsed
            else:
                # Retry with a different endpoint next time.
                time.sleep(0.2)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(0.2)
        except Exception:
            time.sleep(0.2)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_conversation_history(context: str) -> str:
    return context or "No previous conversation."


# ---------------------------------------------------------------------------
# AskMind: non-final turn grading
# - Checklist mode (preferred): uses required_points
# - Legacy mode: judge returns a discrete score
# ---------------------------------------------------------------------------
def _evaluate_non_final_turn_checklist(
    ori_question: str,
    degraded_info: str,
    required_points: List[str],
    current_question: str,
    context: str,
    response: str
) -> float:
    """
    Use required_points as a checklist.
    The judge only marks per-item coverage and whether the assistant answered prematurely.
    This function maps the checklist result to the discrete reward set:
    {-2.0, -0.8, 0.8, 1.0}.
    """
    conversation_history = format_conversation_history(context)

    # Strict JSON schema: avoid letting the judge output an arbitrary score.
    # "hits" must have the same length as required_points, each a boolean.
    judge_prompt = f"""
You will grade an assistant's message in an information-gathering turn of a multi-turn QA.

Original Complete Question:
{ori_question}

Missing/Degraded Info (natural language summary):
{degraded_info}

Required points (the checklist of missing pieces you should evaluate):
{json.dumps(required_points, ensure_ascii=False)}

Previous Conversation:
{conversation_history}

Current User Question:
{current_question}

Assistant's Response:
{response}

Your job:
1) Determine if the assistant prematurely gives a final answer when info is missing (answered_final).
2) For each required point, mark whether the assistant explicitly asks about it (True/False). Do not give credit for vague or unrelated questions.
3) Mark if the assistant asks irrelevant or already-provided info (irrelevant_or_redundant).

Return ONLY the following JSON object (no code fences, no extra text):
{{
  "answered_final": false,
  "hits": [true, false, ...],   // length == len(required_points)
  "irrelevant_or_redundant": false,
  "notes": ["short note per point, same length as hits"]  // keep notes short
}}
"""

    parsed = call_llm_api_json(judge_prompt, max_tokens=16000)
    if parsed is None:
        return DEFAULT_NON_FINAL_FAIL

    try:
        answered_final = bool(parsed.get("answered_final", False))
        hits = parsed.get("hits", [])
        irrelevant = bool(parsed.get("irrelevant_or_redundant", False))

        # Basic validation
        if not isinstance(hits, list) or len(hits) != len(required_points):
            return DEFAULT_NON_FINAL_FAIL

        if answered_final:
            return -2.0

        total = len(hits)
        covered = sum(1 for h in hits if bool(h))

        # Scoring rule (discrete; aligns with the paper's shaping).
        if covered == 0:
            return -0.8
        elif covered == total:
            return 1.0
        else:
            return 0.8

    except Exception:
        return DEFAULT_NON_FINAL_FAIL


def _evaluate_non_final_turn_legacy(
    ori_question: str, degraded_info: str, current_question: str, context: str, response: str
) -> float:
    """
    Legacy mode: when no checklist is available, the judge outputs a discrete score
    (-2.0, -0.8, 0.8, 1.0) which we clamp to the same set.
    """
    conversation_history = format_conversation_history(context)
    prompt = f"""You are evaluating an AI assistant's response in a multi-turn conversation where information is missing.

Original Complete Question (with all information):
{ori_question}

Missing/Degraded Information (what the user didn't provide):
{degraded_info}

Previous Conversation History:
{conversation_history}

Current User Question:
{current_question}

Assistant's Response:
{response}

Scoring Guidelines (choose one):
-2.0: Gives a final answer while required info is missing.
-0.8: Asks questions but misses key missing points or asks irrelevant/redundant info.
 0.8: Asks about at least one key missing point.
 1.0: Asks comprehensively about all/most missing points.

Return ONLY JSON: {{"score": -2.0}} or {{"score": -0.8}} or {{"score": 0.8}} or {{"score": 1.0}}.
"""
    parsed = call_llm_api_json(prompt, max_tokens=16000)
    if parsed is None:
        return DEFAULT_NON_FINAL_FAIL

    try:
        raw = float(parsed.get("score"))
        # Clamp
        if raw <= -1.5:
            return -2.0
        elif raw <= -0.4:
            return -0.8
        elif raw >= 0.9:
            return 1.0
        else:
            return 0.8
    except Exception:
        return DEFAULT_NON_FINAL_FAIL


def evaluate_non_final_turn(
    ori_question: str,
    degraded_info: str,
    current_question: str,
    context: str,
    response: str,
    required_points: Optional[List[str]] = None,
) -> float:
    """Entry point for non-final turn scoring: checklist first, legacy fallback."""
    if not response or not response.strip():
        return -2.0  # Empty response is a critical failure.

    if not degraded_info and not required_points:
        # No missing-info signal to judge against.
        return -2.0

    if required_points and len(required_points) > 0:
        return _evaluate_non_final_turn_checklist(
            ori_question=ori_question,
            degraded_info=degraded_info,
            required_points=required_points,
            current_question=current_question,
            context=context,
            response=response
        )
    else:
        return _evaluate_non_final_turn_legacy(
            ori_question=ori_question,
            degraded_info=degraded_info,
            current_question=current_question,
            context=context,
            response=response
        )


# ---------------------------------------------------------------------------
# Final turn grading (correct / wrong / still asking)
# ---------------------------------------------------------------------------
def evaluate_final_turn(expected_answer: str, current_question: str, context: str, response: str) -> float:
    """
    The judge returns one of three decisions:
    - still_asking -> -2.0
    - wrong       -> -1.0
    - correct     ->  1.0
    """
    if not response or not response.strip():
        return DEFAULT_FINAL_FAIL

    conversation_history = format_conversation_history(context)

    prompt = f"""You evaluate the FINAL answer in a multi-turn conversation.

Previous Conversation:
{conversation_history}

Current User Question (final answer is requested):
{current_question}

Expected Correct Answer (semantic equivalence is acceptable):
{expected_answer}

Assistant's Response:
{response}

Decide one of:
- "still_asking" if the assistant is still asking for info instead of answering.
- "wrong" if the assistant answers but it's incorrect/off-topic/contradicts expected answer.
- "correct" if the assistant's core conclusion matches the expected answer (minor wording differences allowed).

Return ONLY JSON like: {{"decision": "still_asking"}} or {{"decision": "wrong"}} or {{"decision": "correct"}}.
"""
    parsed = call_llm_api_json(prompt, max_tokens=16000)
    if parsed is None:
        return DEFAULT_FINAL_FAIL

    try:
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision == "still_asking":
            return -2.0
        elif decision == "wrong":
            return -1.0
        elif decision == "correct":
            return 1.0
        else:
            return DEFAULT_FINAL_FAIL
    except Exception:
        return DEFAULT_FINAL_FAIL


# ---------------------------------------------------------------------------
# Entry point for VERL integration
# ---------------------------------------------------------------------------
def compute_score_ask_mind_qa(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """
    Multi-turn QA reward function for the AskMind (intent-deficient) dimension.

    - Non-final turn: checklist-based shaping via required_points (preferred);
      falls back to a legacy discrete score otherwise.
    - Final turn: judge correctness only (still_asking/wrong/correct).

    Expected fields in extra_info:
      - is_final_turn: bool
      - ori_question: str
      - degraded_info: str
      - required_points: Optional[List[str]]
      - expected_answer: str (final-turn target; falls back to ground_truth)
      - question: str (current user turn)
      - context: str (conversation history)
    """
    try:
        extra_info = extra_info or {}

        is_final_turn = extra_info.get("is_final_turn", True)
        ori_question = extra_info.get("ori_question", "")
        degraded_info = extra_info.get("degraded_info", "")
        expected_answer = extra_info.get("expected_answer", "")
        current_question = extra_info.get("question", "")
        context = extra_info.get("context", "")
        required_points = extra_info.get("required_points", None)  # Optional: list[str]

        # Empty response: critical failure.
        if not solution_str or not solution_str.strip():
            return -2.0 if not is_final_turn else DEFAULT_FINAL_FAIL

        if is_final_turn:
            if not expected_answer:
                # Fallback to ground_truth if no expected_answer is provided.
                expected_answer = ground_truth
            return evaluate_final_turn(
                expected_answer=expected_answer,
                current_question=current_question,
                context=context,
                response=solution_str
            )

        # Non-final turn.
        return evaluate_non_final_turn(
            ori_question=ori_question,
            degraded_info=degraded_info,
            current_question=current_question,
            context=context,
            response=solution_str,
            required_points=required_points
        )

    except Exception as e:
        # Conservative defaults on any exception (do not return 0.0).
        import traceback; print(f"[compute_score_ask_mind_qa] Error: {e}"); traceback.print_exc()
        is_final_turn = bool((extra_info or {}).get("is_final_turn", True))
        return DEFAULT_FINAL_FAIL if is_final_turn else DEFAULT_NON_FINAL_FAIL


# Optional backwards-compatible alias.
compute_score_multiturn_qa = compute_score_ask_mind_qa


# ---------------------------------------------------------------------------
# Local smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Configured API URLs:", API_URLS)

    print("\n" + "="*50)
    print("Testing Non-Final Turn (Checklist)...")
    print("="*50)
    test_score = evaluate_non_final_turn(
        ori_question="Count letters in a string, report total counts and longest runs.",
        degraded_info="Missing exact counts of A/B/C; definition of 'long same letters in a row' ambiguous; examples may not represent the full set.",
        current_question="I want to analyze this string.",
        context="",
        response="Could you clarify what 'long same letters in a row' means exactly? Also, should I give the count for each letter like A/B/C?",
        required_points=[
            "Exact count of each letter (A, B, C) is missing",
            "Meaning of 'long same letters in a row' is ambiguous",
            "Unclear if examples are representative of the full set"
        ]
    )
    print(f"Checklist test score (expect 0.8 or 1.0 depending on judge): {test_score}")

    print("\n" + "="*50)
    print("Testing Non-Final Turn (Legacy)...")
    print("="*50)
    test_score = evaluate_non_final_turn(
        ori_question="Solve for x: 2x + 3 = 11, x must be a positive integer.",
        degraded_info="The constraint that x must be a positive integer is missing.",
        current_question="Solve for x: 2x + 3 = 11",
        context="",
        response="Do we require x to be an integer and positive?"
    )
    print(f"Legacy test score: {test_score}")

    print("\n" + "="*50)
    print("Testing Final Turn...")
    print("="*50)
    test_score = evaluate_final_turn(
        expected_answer="x = 4",
        current_question="Now give me the final answer.",
        context="User: Solve for x: 2x + 3 = 11\nAssistant: Are there any constraints?\nUser: x must be a positive integer.",
        response="x = 4."
    )
    print(f"Final turn score (expect 1.0): {test_score}")
