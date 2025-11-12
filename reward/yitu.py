# verl/utils/reward_score/multiturn_qa.py
import os
import random
import time
from typing import List, Tuple, Optional, Dict, Any
import requests
import json
import re

# -------------------------
# 配置：API 端点与默认分数
# -------------------------
API_URLS = [
    "http://10.80.13.230:8012/v1/chat/completions",
    "http://10.80.13.230:8013/v1/chat/completions",
    "http://10.80.13.230:8014/v1/chat/completions",
    "http://10.80.13.230:8015/v1/chat/completions",
    # "http://10.80.13.117:8012/v1/chat/completions",
    # "http://10.80.13.117:8013/v1/chat/completions"
]

# 相关代码我已经注释掉了，不用管ENV_FILE_PATH的值，只要配置API_URLS即可
ENV_FILE_PATH = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhouyang/verl_v5/.env"

# 统一的失败默认（不再返回 0.0）
DEFAULT_NON_FINAL_FAIL = -0.8  # 非最终轮：解析失败/异常时的保守负分
DEFAULT_FINAL_FAIL = -1.0      # 最终轮：解析失败/异常时的保守负分


# -------------------------
# 工具：加载端点 & 解析 JSON
# -------------------------
def load_api_urls_from_env(env_path: str = ENV_FILE_PATH, max_retries: int = 3, retry_delay: float = 1.0) -> List[str]:
    """从 .env 文件读取 VLLM_BASE_URL（多行、逗号分隔均可），补齐 /v1/chat/completions"""
    # for attempt in range(max_retries):
    #     try:
    #         if not os.path.exists(env_path):
    #             return API_URLS

    #         with open(env_path, 'r') as f:
    #             content = f.read()

    #         pattern = r'VLLM_BASE_URL\s*=\s*["\']([^"\']*(?:\n[^"\']*)*)["\']'
    #         match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    #         if match:
    #             url_content = match.group(1).strip()
    #             url_content = re.sub(r'\s+', '', url_content)  # 去掉所有空白
    #             urls = [u.strip() for u in url_content.split(',') if u.strip()]

    #             complete_urls = []
    #             for url in urls:
    #                 if not url:
    #                     continue
    #                 url = url.rstrip('/')
    #                 if url.endswith('/v1/chat/completions'):
    #                     complete_urls.append(url)
    #                 elif url.endswith('/v1'):
    #                     complete_urls.append(url + '/chat/completions')
    #                 elif url.endswith('/chat/completions'):
    #                     complete_urls.append(url.replace('/chat/completions', '/v1/chat/completions'))
    #                 else:
    #                     complete_urls.append(url + '/v1/chat/completions')
    #             if complete_urls:
    #                 return complete_urls

    #         return API_URLS

    #     except OSError:
    #         if attempt < max_retries - 1:
    #             time.sleep(retry_delay)
    #             env_path = os.path.abspath(env_path)
    #             continue
    #         else:
    #             import traceback; traceback.print_exc()
    #             return API_URLS
    #     except Exception:
    #         import traceback; traceback.print_exc()
    #         return API_URLS
    return API_URLS


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    容错解析：优先提取 ```json ... ``` 块；否则尝试整体 json.loads；
    若结尾有多余字符，尝试截取首尾大括号做解析。
    """
    if not text:
        return None

    # 1) 代码块
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass

    # 2) 直接 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3) 截取第一个 { 到最后一个 } 之间
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    return None


# -------------------------
# Chat 裁判：JSON-only
# -------------------------
def call_llm_api_json(prompt: str, max_retries: int = 10, max_tokens: int = 16000) -> Optional[Dict[str, Any]]:
    """
    调用裁判模型，要求只返回 JSON。内部做多端点重试和 JSON 容错解析。
    失败返回 None，由上层负责给默认分。
    """
    for attempt in range(max_retries):
        try:
            api_urls = load_api_urls_from_env()
            if not api_urls:
                time.sleep(0.3)
                continue

            api_url = random.choice(api_urls)
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "grader",
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
                # 下一次重试换端点
                time.sleep(0.2)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(0.2)
        except Exception:
            time.sleep(0.2)
    return None


# -------------------------
# 文本整理
# -------------------------
def format_conversation_history(context: str) -> str:
    return context or "No previous conversation."


# -------------------------
# 非最终轮：两种模式
# 1) checklist 模式（优先，使用 required_points）
# 2) 兼容老模式（无 required_points，用离散档位直接让裁判打分）
# -------------------------
def _evaluate_non_final_turn_checklist(
    ori_question: str,
    degraded_info: str,
    required_points: List[str],
    current_question: str,
    context: str,
    response: str
) -> float:
    """
    使用 required_points 作为 checklist。裁判仅负责判断逐点命中与是否提前给出最终答案；
    分数在本函数中依据命中率/行为规则进行归并：{-2.0, -0.8, 0.8, 1.0}
    """
    conversation_history = format_conversation_history(context)

    # 强约束 JSON Schema（不让裁判直接给 score，避免跑偏）
    # hits 必须和 required_points 等长，每个元素 True/False
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

        # 基础合法性检查
        if not isinstance(hits, list) or len(hits) != len(required_points):
            return DEFAULT_NON_FINAL_FAIL

        if answered_final:
            return -2.0

        total = len(hits)
        covered = sum(1 for h in hits if bool(h))

        # 评分规则（可按需微调）
        if covered == 0:
            # 一个点都没问到 → 差
            return -0.8
        elif covered == total:
            # 全覆盖 → 优
            return 1.0
        else:
            # 部分覆盖 → 好
            # 如果明显问了很多无关问题，也可在此降级为 -0.8（视需求决定）
            return 0.8

    except Exception:
        return DEFAULT_NON_FINAL_FAIL


def _evaluate_non_final_turn_legacy(
    ori_question: str, degraded_info: str, current_question: str, context: str, response: str
) -> float:
    """
    兼容老版：无 checklist 时，让裁判直接输出离散分（-2.0, -0.8, 0.8, 1.0），我们再做钳制。
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
        # 钳制
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
    """非最终轮评分入口：优先使用 checklist；无 checklist 时回退到老方式"""
    if not response or not response.strip():
        return -2.0  # 空响应 = 关键失败

    if not degraded_info and not required_points:
        # 没有缺口信息无法评审，判关键失败
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


# -------------------------
# 最终轮：JSON 评审
# -------------------------
def evaluate_final_turn(expected_answer: str, current_question: str, context: str, response: str) -> float:
    """
    让裁判只返回三类决策：
      - still_asking → -2.0
      - wrong        → -1.0
      - correct      →  1.0
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


# -------------------------
# 顶层：统一入口（支持 required_points）
# -------------------------
def compute_score_medical_qa(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """
    通用多轮 QA 奖励函数：
      - 非最终轮：根据 required_points checklist 逐点命中给分；若无 checklist，走兼容老规则。
      - 最终轮：只看是否还在追问/对错。
    """
    try:
        extra_info = extra_info or {}

        is_final_turn = extra_info.get("is_final_turn", True)
        ori_question = extra_info.get("ori_question", "")
        degraded_info = extra_info.get("degraded_info", "")
        expected_answer = extra_info.get("expected_answer", "")
        current_question = extra_info.get("question", "")
        context = extra_info.get("context", "")
        required_points = extra_info.get("required_points", None)  # 新字段：list[str]

        # 空响应直接关键失败
        if not solution_str or not solution_str.strip():
            return -2.0 if not is_final_turn else DEFAULT_FINAL_FAIL

        if is_final_turn:
            if not expected_answer:
                # 回退 ground_truth
                expected_answer = ground_truth
            return evaluate_final_turn(
                expected_answer=expected_answer,
                current_question=current_question,
                context=context,
                response=solution_str
            )

        # 非最终轮
        return evaluate_non_final_turn(
            ori_question=ori_question,
            degraded_info=degraded_info,
            current_question=current_question,
            context=context,
            response=solution_str,
            required_points=required_points
        )

    except Exception as e:
        # 统一失败默认（不再返回 0.0）
        import traceback; print(f"[compute_score_medical_qa] Error: {e}"); traceback.print_exc()
        is_final_turn = bool((extra_info or {}).get("is_final_turn", True))
        return DEFAULT_FINAL_FAIL if is_final_turn else DEFAULT_NON_FINAL_FAIL


# 向后兼容别名
compute_score_multiturn_qa = compute_score_medical_qa


# -------------------------
# 自测
# -------------------------
if __name__ == "__main__":
    print("Testing API URLs loading...")
    res = load_api_urls_from_env()
    print("Loaded API URLs:", res)

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
