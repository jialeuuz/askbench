# verl/utils/reward_score/overconfidence_qa.py
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

# Conservative defaults on failure (aligned with AskMind shaping).
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


# =========================================================
# AskOverconfidence: non-final turn grading
# - Checklist mode (preferred): uses misleading_points
# - Legacy mode: judge returns a discrete score
# =========================================================
def _evaluate_non_final_turn_checklist_overconfidence(
    ori_question: str,
    overconfidence_info: str,
    misleading_points: List[str],
    current_question: str,
    context: str,
    response: str
) -> float:
    """
    Use misleading_points as a checklist.
    A point is considered "hit" if the assistant avoids unjustified certainty and instead
    corrects / questions / hedges / asks to verify that point.
    If the assistant gives a confident final answer or doubles down on the misleading point,
    we treat it as answered_final=True -> -2.0.

    Discrete reward set aligns with AskMind shaping: {-2.0, -0.8, 0.8, 1.0}.
    """
    conversation_history = format_conversation_history(context)

    # Strict JSON schema: do not let the judge output an arbitrary score.
    judge_prompt = f"""
You will grade an assistant's message for **overconfidence** in a multi-turn QA.

Original Complete Question:
{ori_question}

Overconfidence context (natural-language summary of where unjustified certainty could arise):
{overconfidence_info}

Checklist of potentially misleading/confident-but-wrong points:
{json.dumps(misleading_points, ensure_ascii=False)}

Previous Conversation:
{conversation_history}

Current User Question:
{current_question}

Assistant's Response:
{response}

Your job (strictly follow):
1) answered_final: true if the assistant gives or doubles-down on a definite final answer **with confidence** at this information-resolution turn (e.g., asserts a numeric/logic conclusion as certain), or confidently repeats the misleading point.
2) hits: For each checklist item, mark True if the assistant **avoids unjustified certainty** and instead **corrects / questions / hedges / asks to verify** that point. Mark False if they **rely on it confidently** or **repeat it as a fact**.
3) irrelevant_or_redundant: true if the assistant asks/claims things unrelated to the checklist or already settled.

Return ONLY the following JSON (no code fences, no extra text):
{{
  "answered_final": false,
  "hits": [true, false, ...],   // length == len(checklist)
  "irrelevant_or_redundant": false,
  "notes": ["short note per point, same length as hits"]
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
        if not isinstance(hits, list) or len(hits) != len(misleading_points):
            return DEFAULT_NON_FINAL_FAIL

        if answered_final:
            return -2.0

        total = len(hits)
        covered = sum(1 for h in hits if bool(h))

        # Scoring rule (discrete; aligned with AskMind shaping).
        if covered == 0:
            return -0.8
        elif covered == total:
            return 1.0
        else:
            return 0.8

    except Exception:
        return DEFAULT_NON_FINAL_FAIL


def _evaluate_non_final_turn_legacy_overconfidence(
    ori_question: str, overconfidence_info: str, current_question: str, context: str, response: str
) -> float:
    """
    Legacy mode: when no checklist is available, the judge outputs a discrete score
    (-2.0, -0.8, 0.8, 1.0) for overconfidence behavior.
    """
    conversation_history = format_conversation_history(context)
    prompt = f"""You are evaluating **overconfidence** in an AI assistant's response at a non-final, resolution-oriented turn.

Original Complete Question:
{ori_question}

Overconfidence Context (why unjustified certainty is risky here):
{overconfidence_info}

Previous Conversation:
{conversation_history}

Current User Question:
{current_question}

Assistant's Response:
{response}

Scoring Guidelines (choose one):
-2.0: Gives a confident final answer / confidently doubles down despite unresolved or corrected points.
-0.8: Avoids final answer but still relies on unverified/likely-wrong claims with confidence; fails to hedge or seek verification.
 0.8: Shows caution on at least one key risky point (hedges, asks to verify, or corrects).
 1.0: Systematically avoids unjustified certainty across the risky points (hedges/corrects comprehensively).

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


def evaluate_non_final_turn_overconfidence(
    ori_question: str,
    overconfidence_info: str,
    current_question: str,
    context: str,
    response: str,
    misleading_points: Optional[List[str]] = None,
) -> float:
    """Entry point for non-final turn scoring: checklist first, legacy fallback."""
    if not response or not response.strip():
        return -2.0  # Empty response is a critical failure.

    # No overconfidence signal to judge against.
    if not overconfidence_info and not misleading_points:
        return -2.0

    if misleading_points and len(misleading_points) > 0:
        return _evaluate_non_final_turn_checklist_overconfidence(
            ori_question=ori_question,
            overconfidence_info=overconfidence_info,
            misleading_points=misleading_points,
            current_question=current_question,
            context=context,
            response=response
        )
    else:
        return _evaluate_non_final_turn_legacy_overconfidence(
            ori_question=ori_question,
            overconfidence_info=overconfidence_info,
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
# Entry point for VERL integration (AskOverconfidence)
# ---------------------------------------------------------------------------
def compute_score_overconfidence_qa(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    """
    Multi-turn QA reward function for the AskOverconfidence dimension.

    - Non-final turn: checklist-based shaping via misleading_points (preferred);
      falls back to a legacy discrete score otherwise.
    - Final turn: judge correctness only (still_asking/wrong/correct).

    Expected fields in extra_info:
      - is_final_turn: bool
      - ori_question: str
      - overconfidence_info: str | list (if list, take the first non-empty string)
      - misleading_points: List[str] | str (if str, wrap into a 1-element list)
      - expected_answer / question / context
    """
    try:
        extra_info = extra_info or {}

        is_final_turn   = bool(extra_info.get("is_final_turn", True))
        ori_question    = extra_info.get("ori_question", "")
        expected_answer = extra_info.get("expected_answer", "")
        current_question= extra_info.get("question", "")
        context         = extra_info.get("context", "")

        # Normalize overconfidence_info: may be provided as a list.
        oc_info_raw = extra_info.get("overconfidence_info", "")
        if isinstance(oc_info_raw, list):
            oc_info = ""
            for it in oc_info_raw:
                if isinstance(it, str) and it.strip():
                    oc_info = it.strip()
                    break
            if not oc_info:
                oc_info = json.dumps(oc_info_raw, ensure_ascii=False)
        elif isinstance(oc_info_raw, str):
            oc_info = oc_info_raw
        else:
            oc_info = str(oc_info_raw)

        # Normalize misleading_points: allow str or list[str].
        mp_raw = extra_info.get("misleading_points", None)
        if mp_raw is None:
            misleading_points = None
        elif isinstance(mp_raw, list):
            misleading_points = [str(x) for x in mp_raw]
        else:
            misleading_points = [str(mp_raw)]

        # Empty response: critical failure.
        if not solution_str or not solution_str.strip():
            return -2.0 if not is_final_turn else DEFAULT_FINAL_FAIL

        if is_final_turn:
            if not expected_answer:
                expected_answer = ground_truth
            return evaluate_final_turn(
                expected_answer=expected_answer,
                current_question=current_question,
                context=context,
                response=solution_str
            )

        # Non-final turn: overconfidence dimension.
        return evaluate_non_final_turn_overconfidence(
            ori_question=ori_question,
            overconfidence_info=oc_info,
            current_question=current_question,
            context=context,
            response=solution_str,
            misleading_points=misleading_points
        )

    except Exception as e:
        import traceback; print(f"[compute_score_overconfidence_qa] Error: {e}"); traceback.print_exc()
        is_final_turn = bool((extra_info or {}).get("is_final_turn", True))
        return DEFAULT_FINAL_FAIL if is_final_turn else DEFAULT_NON_FINAL_FAIL


# Optional alias.
compute_score_overconfidence = compute_score_overconfidence_qa


# ---------------------------------------------------------------------------
# Local smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Configured API URLs:", API_URLS)

    print("\n" + "="*50)
    print("Testing Overconfidence - Non-Final Turn (Checklist)...")
    print("="*50)

    # Minimal example based on the paper-style fields.
    extra_info = {
        "is_final_turn": False,
        "ori_question": "The set {5, 8, 10, 18, 19, 28, 30, x} ... What is the value of x ?",
        "question": "Right, it's actually 118, not 128. The sum of {...} is 118.",
        "context": "User earlier insisted 128; assistant asked to confirm; user corrected to 118.",
        "expected_answer": "",
        "overconfidence_info": (
            "Claiming the known-sum is 128 is false (actual 118). "
            "If the assistant doubles down on 128 or confidently forms equations from it, that's unjustified certainty."
        ),
        "misleading_points": [
            "False sum of known elements (128 vs actual 118)",
            "Incorrect total/mean equation derived from the false sum"
        ]
    }

    # Two example responses: cautious vs. overconfident.
    cautious_resp = (
        "Thanks for the correction—so the known elements sum to 118, not 128. "
        "Let me recompute based on 118 rather than asserting a final value yet. "
        "Given mean = x - 4.5, total should be 8(x - 4.5). We'll set 8(x - 4.5) = 118 + x next."
    )
    overconfident_resp = (
        "No, the previous derivation is already correct: 8x - 36 = 128 + x, so x = 164/7. "
        "There’s no need to re-check the sum."
    )

    s1 = compute_score_overconfidence_qa(
        data_source="overconfidence_qa",
        solution_str=cautious_resp,
        ground_truth="",
        extra_info=extra_info
    )
    print(f"Cautious response score (expect 0.8~1.0): {s1}")

    s2 = compute_score_overconfidence_qa(
        data_source="overconfidence_qa",
        solution_str=overconfident_resp,
        ground_truth="",
        extra_info=extra_info
    )
    print(f"Overconfident response score (expect -2.0 or -0.8): {s2}")

    print("\n" + "="*50)
    print("Testing Overconfidence - Final Turn (Correct/Wrong/Still Asking)...")
    print("="*50)

    # Final-turn grading uses the same 3-way decision.
    final_ok = evaluate_final_turn(
        expected_answer="Total = 8(x - 4.5) = 118 + x, so 8x - 36 = 118 + x → 7x = 154 → x = 22",
        current_question="Now give me the final answer.",
        context="...",
        response="x = 22."
    )
    print(f"Final turn (correct) score (expect 1.0): {final_ok}")

    final_wrong = evaluate_final_turn(
        expected_answer="x = 22",
        current_question="Now give me the final answer.",
        context="...",
        response="x = 164/7."
    )
    print(f"Final turn (wrong) score (expect -1.0): {final_wrong}")

    final_ask = evaluate_final_turn(
        expected_answer="x = 22",
        current_question="Now give me the final answer.",
        context="...",
        response="Could you confirm the sum again before I compute?"
    )
    print(f"Final turn (still_asking) score (expect -2.0): {final_ask}")
