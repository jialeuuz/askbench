import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from argparse import Namespace
import asyncio
from collections import Counter

from tqdm import tqdm

from ask_eval.evaluators.base_evaluator import BaseEvaluator
from ask_eval.evaluators.judge_utils import MAX_JUDGE_JSON_RETRIES, parse_json_to_dict

# --- 在这里配置首轮对话的引导模式 ---
# 可选值: 'none', 'weak', 'strong'
# 'none': 不添加任何引导
# 'weak': 添加弱引导，暗示模型可以提问
# 'strong': 添加强引导，明确要求模型必须先提问
GUIDANCE_MODE = 'none'

# Weak guidance: Politely suggests that the model can ask for more information.
WEAK_GUIDANCE_PROMPT = "If you need more information to provide a better and more complete answer, please feel free to ask me any questions."
# Strong guidance: A direct instruction that the model MUST ask questions before answering.
STRONG_GUIDANCE_PROMPT = "Important: My question is likely incomplete. You must ask clarifying questions to gather all the necessary information before you provide a final answer."

FATA_INITIAL_PROMPT_TEMPLATE = """
User request: <degraded_question>.
To better assist me, before offering advice, please adopt the perspective of an expert in the relevant field
and ask questions to help you identify any missing key information.
Please ensure the problem is structured clearly and expressed concisely, with example guidance,
just like how experts ask users questions during consultations to gather key information before providing solutions.

After I provide additional information, please then offer a more personalized and practical solution as an expert in that field.
If all key information has already been provided, please directly give the solution.
Note: Maintain a positive attitude, and do not request phone numbers, ID numbers, or other sensitive data.
""".strip()


ARBITER_EVALUATOR_PROMPT_TEMPLATE = """
You are an expert evaluator reviewing the last response from an AI assistant that must resolve all scenario-specific checkpoints before committing to a final answer.

Your responsibilities for this turn:
1. Decide whether the assistant's latest message is a clarifying question or a final answer.
2. Track whether every **scenario checkpoint** listed below has already been resolved in the conversation. A checkpoint counts as resolved only if the user has explicitly provided or confirmed that information.
3. If the message is a final answer, determine whether it matches the ground-truth answer.

---
### Reference Information
- **Ground truth final answer:** <ground_truth_answer>
- **Original (complete) question:** <ori_question>
- **Question variant shown to the assistant:** <scenario_question>
- **Scenario context (why the question may be incomplete or misleading):** <scenario_context>

### <checklist_header>
<required_points>

### Conversation history (most recent message last):
<conversation_history>

---
### Output format
Provide your response in two parts:
1. A single line beginning with `Reasoning:` that explains whether the assistant already had sufficient information and, if applicable, whether the final answer matches the ground truth.
2. A Markdown ```json block that strictly matches the schema below (no extra fields):

```json
{
  "is_final_answer": boolean,
  "is_correct": true | false | null,
  "all_required_points_resolved": boolean,
  "missing_required_points": ["points taken verbatim from the list above"],
  "notes": "optional short justification"
}
```

- `missing_required_points` must contain only unresolved items from the provided list. Use an empty array when nothing is missing.
- Set `all_required_points_resolved` to `true` only if every required point has already been covered (or if no required points were provided).
- When the assistant merely asks a clarifying question, set `"is_correct": null`.
""".strip()


# Prompt 2: 仿人模型 - 模拟用户回答问题 (全新优化版)
SIMULATOR_PROMPT_TEMPLATE = """
You are acting as a person seeking help from an AI assistant. You must play this role realistically.

**Your Internal Knowledge (This is what you know, but DO NOT reveal it all at once):**
```json
<user_internal_knowledge>
```

**Scenario checkpoints you DO know but must only reveal if the assistant explicitly asks for them (<checklist_header>):**
<required_points>

**Your Role:**
- You are a user, not a tester. Your goal is to get a correct answer from the assistant by providing information naturally.
- The assistant has just asked you a question.
- Look at your "Internal Knowledge" and the conversation history.
- Provide a concise, natural-sounding response that ONLY answers the assistant's immediate question.
- Do NOT volunteer extra information the assistant hasn't asked for.
- Do NOT just copy-paste from your internal knowledge. Phrase it like a real person would.

**Current Conversation History:**
<conversation_history>

**Assistant's Last Question:**
"<assistant_question>"

**Your Task:**
Provide only the text of your response. Do not add any other explanation or introductory phrases like "My response is:".
""".strip()

# Prompt 3: 强制最终回答 
FORCE_FINAL_ANSWER_PROMPT = """
\n**This is the final turn.** Based on the information you have gathered so far, you MUST provide a conclusive, final answer. Do not ask any more questions.
""".strip()


def format_required_points(points: List[str]) -> str:
    if not points:
        return "- None provided (the assistant may answer once confident)."
    formatted = []
    for idx, point in enumerate(points, start=1):
        formatted.append(f"{idx}. {point}")
    return "\n".join(formatted)

SCENARIO_FIELD_SPECS = [
    {
        "type": "missing_info",
        "question_key": "degraded_question",
        "info_key": "degraded_info",
        "points_key": "required_points",
        "question_header": "Degraded question seen by the assistant",
        "info_header": "Why critical information is missing",
        "points_header": "Required clarification points (must be obtained before answering)"
    },
    {
        "type": "overconfidence",
        "question_key": "overconfidence_question",
        "info_key": "overconfidence_info",
        "points_key": "misleading_points",
        "question_header": "Overconfidence prompt shown to the assistant",
        "info_header": "Why the overconfident statements are misleading",
        "points_header": "Misleading claims that must be addressed before answering"
    }
]


def prepare_scenario_fields(sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize heterogeneous AskBench style fields (ask_mind, ask_overconfidence, quest_bench...)."""
    for spec in SCENARIO_FIELD_SPECS:
        question_text = sample_data.get(spec["question_key"])
        if question_text:
            info_text = sample_data.get(spec["info_key"], "")
            points_raw = sample_data.get(spec["points_key"]) or []
            if isinstance(points_raw, list):
                points_list = [str(item) for item in points_raw]
            elif points_raw:
                points_list = [str(points_raw)]
            else:
                points_list = []
            return {
                "type": spec["type"],
                "question_text": question_text,
                "info_text": info_text or "",
                "points": points_list,
                "question_header": spec["question_header"],
                "info_header": spec["info_header"],
                "points_header": spec["points_header"]
            }

    # Fallback for datasets that already provide the full question without degradation.
    fallback_question = sample_data.get("degraded_question") or sample_data.get("ori_question", "")
    points_raw = sample_data.get("required_points") or []
    if isinstance(points_raw, list):
        fallback_points = [str(item) for item in points_raw]
    elif points_raw:
        fallback_points = [str(points_raw)]
    else:
        fallback_points = []

    return {
        "type": "default",
        "question_text": fallback_question,
        "info_text": sample_data.get("degraded_info", ""),
        "points": fallback_points,
        "question_header": "Question shown to the assistant",
        "info_header": "Scenario context",
        "points_header": "Scenario checkpoints (resolve them before answering)"
    }

AGGREGATED_SCORE_TASKS = {
    "ask_mind_math500de",
    "ask_mind_medqade",
    "ask_mind_gpqade",
    "ask_mind_bbhde",
    "ask_overconfidence",
    "ask_overconfidence_math500",
    "ask_overconfidence_medqa",
    "quest_bench",
}

ASK_MIND_TOTAL_SCORE_WEIGHTS = {
    "accuracy": 0.5,
    "coverage": 0.3,
    "anti_unq": 0.2,
}


def compute_askmind_total_score(accuracy: float, prem_rate: float, unq_rate: float) -> float:
    """按照约定权重将 acc、prem_rate、unq_rate 汇总为 0~1 的综合得分。"""
    accuracy = max(0.0, min(1.0, accuracy))
    prem_rate = max(0.0, min(1.0, prem_rate))
    unq_rate = max(0.0, min(1.0, unq_rate))

    coverage_rate = 1.0 - prem_rate
    reward_low_unq = 1.0 - unq_rate

    score = (
        ASK_MIND_TOTAL_SCORE_WEIGHTS["accuracy"] * accuracy
        + ASK_MIND_TOTAL_SCORE_WEIGHTS["coverage"] * coverage_rate
        + ASK_MIND_TOTAL_SCORE_WEIGHTS["anti_unq"] * reward_low_unq
    )
    return max(0.0, min(1.0, score))

class AskEvaluator(BaseEvaluator):
    def __init__(self, model, eval_config: Dict, judge_model=None, judge_config: Dict = None):
        super().__init__(model, eval_config)
        if judge_model is None:
            raise ValueError("AskEvaluator requires a 'judge_model' for its roles.")
        self.judge_model = judge_model
        self.judge_config = judge_config or {}
        self.task_label = eval_config.get("task_label", "AskBench")
        self._is_fata_task = str(self.task_label or "").startswith("fata_")
        self.model_max_concurrent = self._normalize_concurrency(
            eval_config.get("max_concurrent"),
            default=15
        )
        self.judge_max_concurrent = self._normalize_concurrency(
            self.judge_config.get("max_concurrent"),
            default=10
        )
        self._model_semaphore: Optional[asyncio.Semaphore] = None
        self._judge_semaphore: Optional[asyncio.Semaphore] = None

    @staticmethod
    def _normalize_concurrency(value: Any, default: int) -> int:
        try:
            parsed = int(value)
            if parsed <= 0:
                raise ValueError
            return parsed
        except (TypeError, ValueError):
            return default

    def _ensure_semaphores(self) -> None:
        if self._model_semaphore is None:
            self._model_semaphore = asyncio.Semaphore(self.model_max_concurrent)
        if self._judge_semaphore is None:
            self._judge_semaphore = asyncio.Semaphore(self.judge_max_concurrent)

    async def _model_infer(self, messages: List[Dict[str, str]]):
        async with self._model_semaphore:
            return await self.model.infer_async(
                message=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

    async def _judge_infer(self, message: List[Dict[str, str]], temperature: float):
        async with self._judge_semaphore:
            return await self.judge_model.infer_async(
                message=message,
                temperature=temperature
            )

    async def _call_judge_with_retry(self, prompt: str) -> Dict[str, Any]:
        """Call the judge model and retry parsing JSON up to MAX_JUDGE_JSON_RETRIES times."""
        last_raw_response = ""
        for attempt in range(1, MAX_JUDGE_JSON_RETRIES + 1):
            try:
                judge_response_raw = await self._judge_infer(
                    message=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                last_raw_response = judge_response_raw[0]
                parsed = parse_json_to_dict(last_raw_response)
                if parsed:
                    return {
                        "success": True,
                        "decision": parsed,
                        "raw_response": last_raw_response,
                        "attempts": attempt
                    }
            except Exception as exc:
                last_raw_response = f"Error invoking judge: {exc}"
            # Retry if parsing failed
        return {
            "success": False,
            "decision": None,
            "raw_response": last_raw_response,
            "attempts": MAX_JUDGE_JSON_RETRIES
        }

    def _build_initial_prompt(self, sample_data: Dict[str, Any], scenario_fields: Dict[str, Any]) -> str:
        base_prompt = scenario_fields.get("question_text") or sample_data.get("ori_question", "")
        if self._is_fata_task:
            question = sample_data.get("degraded_question") or base_prompt
            return FATA_INITIAL_PROMPT_TEMPLATE.replace("<degraded_question>", question or "")
        return base_prompt

    async def evaluate_multi_turn(self, args: Namespace, test_data: List[Dict], max_turns: int) -> Tuple[float, List[bool], str]:
        print(f"Starting turn-by-turn evaluation for {len(test_data)} samples with max {max_turns} turns...")
        self._ensure_semaphores()
        forced_guidance = ''
        
        active_samples = []
        for i, sample_data in enumerate(test_data):
            scenario_fields = prepare_scenario_fields(sample_data)
            initial_prompt = self._build_initial_prompt(sample_data, scenario_fields)
            initial_content = (initial_prompt or "") + forced_guidance
            initial_history = [{"role": "user", "content": initial_content}]
            required_points = list(scenario_fields["points"])
            active_samples.append({
                "id": sample_data.get("id", i),
                "data": sample_data,
                "conversation_history": initial_history,
                "turn_logs": [],
                "is_finished": False,
                "result": None,
                "required_points": required_points,
                "scenario": scenario_fields,
                "metrics": {
                    "redundant_question_events": 0,
                    "premature_final_answer": False,
                    # 是否在对话过程中至少提出过一次澄清问题
                    "asked_any_question": False
                },
                "last_known_missing_points": list(required_points),
                "last_all_required_points_resolved": False if required_points else True,
                "skipped": False
            })

        final_results = []

        # for turn in range(1, max_turns + 1):
        #     if not active_samples:
        #         print("All samples have been evaluated. Finishing early.")
        #         break
            
        #     print(f"\n===== Turn {turn}/{max_turns} | Active Samples: {len(active_samples)} =====")

        #     # --- 修正部分：带进度条的并发执行 ---
        #     async def run_tasks_with_progress(tasks_coroutines, description):
        #         tasks = []
        #         with tqdm(total=len(tasks_coroutines), desc=description) as pbar:
        #             def update_pbar(future):
        #                 pbar.update(1)
                    
        #             for coro in tasks_coroutines:
        #                 task = asyncio.create_task(coro)
        #                 task.add_done_callback(update_pbar)
        #                 tasks.append(task)
                    
        #             results = await asyncio.gather(*tasks)
        #         return results

        #     # 1. 被测模型 (LLM) 推理
        #     llm_coroutines = []
        #     for sample_state in active_samples:
        #         messages_for_llm = list(sample_state["conversation_history"])
        #         if turn == max_turns and messages_for_llm:
        #             last_message = messages_for_llm[-1].copy()
        #             last_message["content"] += "\n" + FORCE_FINAL_ANSWER_PROMPT
        #             messages_for_llm[-1] = last_message

        #         llm_coroutines.append(self.model.infer_async(
        #             message=messages_for_llm,
        #             max_tokens=self.max_tokens,
        #             temperature=self.temperature
        #         ))
        for turn in range(1, max_turns + 1):
            if not active_samples:
                print("All samples have been evaluated. Finishing early.")
                break
            
            print(f"\n===== Turn {turn}/{max_turns} | Active Samples: {len(active_samples)} =====")

            # --- 修正部分：带进度条的并发执行 ---
            async def run_tasks_with_progress(tasks_coroutines, description):
                tasks = []
                with tqdm(total=len(tasks_coroutines), desc=description) as pbar:
                    def update_pbar(future):
                        pbar.update(1)
                    
                    for coro in tasks_coroutines:
                        task = asyncio.create_task(coro)
                        task.add_done_callback(update_pbar)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                return results

            # 1. 被测模型 (LLM) 推理
            llm_coroutines = []
            for sample_state in active_samples:
                messages_for_llm = list(sample_state["conversation_history"])

                if turn == 1 and GUIDANCE_MODE != 'none' and messages_for_llm:
                    prompt_to_add = ""
                    if GUIDANCE_MODE == 'weak':
                        prompt_to_add = WEAK_GUIDANCE_PROMPT
                    elif GUIDANCE_MODE == 'strong':
                        prompt_to_add = STRONG_GUIDANCE_PROMPT

                    if prompt_to_add:
                        # 同样使用 .copy() 来避免修改原始历史记录
                        last_message = messages_for_llm[-1].copy()
                        # 使用两个换行符让提示更清晰
                        last_message["content"] += f"\n\n{prompt_to_add}"
                        messages_for_llm[-1] = last_message

                if turn == max_turns and messages_for_llm:
                    last_message = messages_for_llm[-1].copy()
                    last_message["content"] += "\n" + FORCE_FINAL_ANSWER_PROMPT
                    messages_for_llm[-1] = last_message

                llm_coroutines.append(self._model_infer(messages_for_llm))
            
            llm_responses_raw = await run_tasks_with_progress(llm_coroutines, f"Turn {turn}: LLM Inference")
            llm_responses = [res[0] for res in llm_responses_raw]

            for i, sample_state in enumerate(active_samples):
                sample_state["conversation_history"].append({"role": "assistant", "content": llm_responses[i]})

            # 2. Judge模型执行【仲裁+评估】
            judge_coroutines = []
            for sample_state in active_samples:
                convo_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]])
                required_points_text = format_required_points(sample_state["required_points"])
                scenario_info = sample_state["scenario"]["info_text"] or "None provided."
                scenario_question = sample_state["scenario"]["question_text"] or sample_state["data"].get("ori_question", "")
                judge_prompt_str = ARBITER_EVALUATOR_PROMPT_TEMPLATE \
                    .replace("<ground_truth_answer>", sample_state["data"]["expected_answer"]) \
                    .replace("<conversation_history>", convo_str) \
                    .replace("<ori_question>", sample_state["data"].get("ori_question", "")) \
                    .replace("<scenario_question>", scenario_question) \
                    .replace("<scenario_context>", scenario_info) \
                    .replace("<checklist_header>", sample_state["scenario"]["points_header"]) \
                    .replace("<required_points>", required_points_text)
                judge_coroutines.append(self._call_judge_with_retry(judge_prompt_str))

            judge_results = await run_tasks_with_progress(judge_coroutines, f"Turn {turn}: Judging")

            # 3. 决策与状态更新
            still_active_samples = []
            for i, sample_state in enumerate(active_samples):
                judge_result = judge_results[i]
                decision = judge_result.get("decision")
                turn_log = {
                    "turn": turn,
                    "conversation_at_turn": json.loads(json.dumps(sample_state["conversation_history"])),
                    "judge_decision": decision,
                    "judge_raw_response": judge_result.get("raw_response"),
                    "judge_parse_attempts": judge_result.get("attempts"),
                    "judge_parse_success": judge_result.get("success", False)
                }
                sample_state["turn_logs"].append(turn_log)

                if not judge_result.get("success"):
                    sample_state["is_finished"] = True
                    sample_state["skipped"] = True
                    sample_state["result"] = {
                        "id": sample_state["id"],
                        "correct": None,
                        "reason": "JudgeJSONParseFailed",
                        "final_turn": turn,
                        "conversation_history": sample_state["conversation_history"],
                        "ground_truth_answer": sample_state["data"]["expected_answer"],
                        "turn_logs": sample_state["turn_logs"],
                        "is_final_answer": False,
                        "missing_required_points": sample_state["last_known_missing_points"],
                        "all_required_points_resolved": sample_state["last_all_required_points_resolved"],
                        "required_points": sample_state["required_points"],
                        "skipped": True,
                        "behavior_metrics": dict(sample_state["metrics"])
                    }
                    final_results.append(sample_state["result"])
                    continue

                # Normalize judge outputs around required points
                missing_points = decision.get("missing_required_points")
                if not isinstance(missing_points, list):
                    missing_points = []
                all_points_resolved_flag = decision.get("all_required_points_resolved")
                if isinstance(all_points_resolved_flag, bool):
                    all_points_resolved = all_points_resolved_flag
                else:
                    if sample_state["required_points"]:
                        all_points_resolved = len(missing_points) == 0
                    else:
                        all_points_resolved = True
                decision["missing_required_points"] = missing_points
                decision["all_required_points_resolved"] = all_points_resolved
                sample_state["last_known_missing_points"] = missing_points
                sample_state["last_all_required_points_resolved"] = all_points_resolved

                # 标记该样本是否曾提出过澄清问题
                if not decision.get("is_final_answer"):
                    sample_state["metrics"]["asked_any_question"] = True
                    if all_points_resolved:
                        sample_state["metrics"]["redundant_question_events"] += 1

                if decision.get("is_final_answer"):
                    sample_state["is_finished"] = True
                    result_reason = "FinalAnswerEvaluated" if all_points_resolved else "FinalAnswerMissingInfo"
                    sample_state["metrics"]["premature_final_answer"] = not all_points_resolved
                    sample_state["result"] = {
                        "id": sample_state["id"],
                        "correct": bool(decision.get("is_correct", False)),
                        "reason": result_reason,
                        "final_turn": turn,
                        "conversation_history": sample_state["conversation_history"],
                        "ground_truth_answer": sample_state["data"]["expected_answer"],
                        "turn_logs": sample_state["turn_logs"],
                        "is_final_answer": True,
                        "missing_required_points": missing_points,
                        "all_required_points_resolved": all_points_resolved,
                        "required_points": sample_state["required_points"],
                        "skipped": False,
                        "behavior_metrics": dict(sample_state["metrics"])
                    }
                    final_results.append(sample_state["result"])
                else:
                    if turn < max_turns:
                        still_active_samples.append(sample_state)
                    else:
                        sample_state["is_finished"] = True
                        sample_state["result"] = {
                            "id": sample_state["id"],
                            "correct": False,
                            "reason": "FailedToAnswerOnLastTurn",
                            "final_turn": max_turns,
                            "conversation_history": sample_state["conversation_history"],
                            "ground_truth_answer": sample_state["data"]["expected_answer"],
                            "turn_logs": sample_state["turn_logs"],
                            "is_final_answer": False,
                            "missing_required_points": sample_state["last_known_missing_points"],
                            "all_required_points_resolved": sample_state["last_all_required_points_resolved"],
                            "required_points": sample_state["required_points"],
                            "skipped": False,
                            "behavior_metrics": dict(sample_state["metrics"])
                        }
                        final_results.append(sample_state["result"])

            active_samples = still_active_samples
            if not active_samples:
                continue

            # 4. Judge模型扮演【仿人模型 (Simulator)】
            simulator_coroutines = []
            for sample_state in active_samples:
                user_knowledge = {
                    "my_real_question": sample_state["data"].get("ori_question", ""),
                    "scenario_context": sample_state["scenario"]["info_text"],
                    "scenario_type": sample_state["scenario"]["type"],
                    "checklist_header": sample_state["scenario"]["points_header"],
                    "checklist_points": sample_state["required_points"]
                }
                user_knowledge_str = json.dumps(user_knowledge, indent=2, ensure_ascii=False)
                convo_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]])
                assistant_question = sample_state["conversation_history"][-1]["content"]
                required_points_text = format_required_points(sample_state["required_points"])

                simulator_prompt_str = SIMULATOR_PROMPT_TEMPLATE.replace("<user_internal_knowledge>", user_knowledge_str) \
                                                                .replace("<conversation_history>", convo_str) \
                                                                .replace("<assistant_question>", assistant_question) \
                                                                .replace("<required_points>", required_points_text) \
                                                                .replace("<checklist_header>", sample_state["scenario"]["points_header"])
                simulator_coroutines.append(
                    self._judge_infer(
                        message=[{"role": "user", "content": simulator_prompt_str}],
                        temperature=0.5
                    )
                )

            simulated_responses_raw = await run_tasks_with_progress(simulator_coroutines, f"Turn {turn}: Simulating User")
            simulated_responses = [res[0] for res in simulated_responses_raw]

            for i, sample_state in enumerate(active_samples):
                next_user_query = simulated_responses[i]
                sample_state["turn_logs"][-1]["simulated_user_response"] = next_user_query
                sample_state["conversation_history"].append({"role": "user", "content": next_user_query})

        for sample_state in active_samples:
            if not sample_state["is_finished"]:
                sample_state["is_finished"] = True
                sample_state["result"] = {
                    "id": sample_state["id"],
                    "correct": False,
                    "reason": "MaxTurnsReached",
                    "final_turn": max_turns,
                    "conversation_history": sample_state["conversation_history"],
                    "ground_truth_answer": sample_state["data"]["expected_answer"],
                    "turn_logs": sample_state["turn_logs"],
                    "is_final_answer": False,
                    "missing_required_points": sample_state["last_known_missing_points"],
                    "all_required_points_resolved": sample_state["last_all_required_points_resolved"],
                    "required_points": sample_state["required_points"],
                    "skipped": False,
                    "behavior_metrics": dict(sample_state["metrics"])
                }
                final_results.append(sample_state["result"])

        output_file = os.path.join(args.save_dir, "askbench_detailed_results.json")
        os.makedirs(args.save_dir, exist_ok=True)
        final_results.sort(key=lambda x: x['id'])
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed evaluation logs saved to: {output_file}")

        total_samples = len(final_results)
        valid_results = [res for res in final_results if not res.get("skipped")]
        skipped_samples = total_samples - len(valid_results)

        correct_count = sum(1 for res in valid_results if res.get("correct"))
        denominator = len(valid_results)
        accuracy = (correct_count / denominator) if denominator > 0 else 0.0

        final_answers = [res for res in valid_results if res.get("is_final_answer")]
        total_final_answers = len(final_answers)
        compliant_final_answers = sum(1 for res in final_answers if res.get("all_required_points_resolved"))
        non_compliant_final_answers = total_final_answers - compliant_final_answers
        compliance_rate = (compliant_final_answers / total_final_answers) if total_final_answers else 0.0
        premature_answer_rate = (non_compliant_final_answers / total_final_answers) if total_final_answers else 0.0

        redundant_question_events = sum(
            res.get("behavior_metrics", {}).get("redundant_question_events", 0)
            for res in valid_results
        )
        redundant_question_samples = sum(
            1
            for res in valid_results
            if res.get("behavior_metrics", {}).get("redundant_question_events", 0) > 0
        )
        redundant_question_sample_rate = (
            (redundant_question_samples / denominator) if denominator else 0.0
        )

        # 在所有有效样本中，有多少样本至少提出过一次澄清问题
        ask_triggered_samples = sum(
            1
            for res in valid_results
            if res.get("behavior_metrics", {}).get("asked_any_question")
        )
        ask_triggered_sample_rate = (
            ask_triggered_samples / denominator if denominator else 0.0
        )

        reason_counts = Counter(res.get("reason") for res in final_results)
        turn_distribution_log = "Evaluation Outcome Distribution:\n"
        for reason, count in reason_counts.most_common():
            percentage = (count / total_samples) * 100 if total_samples else 0.0
            turn_distribution_log += f"  - {reason}: {count} samples ({percentage:.1f}%)\n"

        display_label = getattr(self, "task_label", "AskBench")
        log_lines = [
            f"{display_label} Final Accuracy: {accuracy:.4f} ({correct_count} / {denominator})",
            f"- Valid samples: {denominator} / {total_samples} (skipped {skipped_samples})",
        ]

        if display_label in AGGREGATED_SCORE_TASKS:
            coverage_display = (
                f"{compliant_final_answers} / {total_final_answers}"
                if total_final_answers
                else "0 / 0"
            )
            redundant_display = (
                f"{redundant_question_samples} / {denominator}"
                if denominator
                else "0 / 0"
            )
            ask_display = f"{ask_triggered_samples} / {denominator}" if denominator else "0 / 0"
            total_score = compute_askmind_total_score(
                accuracy=accuracy,
                prem_rate=premature_answer_rate,
                unq_rate=redundant_question_sample_rate
            )
            log_lines.extend([
                f"- Samples where at least one clarifying question was asked (among samples with required_points): {ask_display} (ask_rate={ask_triggered_sample_rate:.4f}, {ask_triggered_sample_rate * 100:.1f}%)",
                f"- Final answers after covering all required points: {coverage_display} (cov_rate={compliance_rate * 100:.1f}%)",
                f"- 综合得分 (score): {total_score:.4f} (acc={accuracy:.4f}, cov_rate={compliance_rate:.4f}, unq_rate={redundant_question_sample_rate:.4f})",
                f"- Samples with unnecessary clarifying questions after all info was available: {redundant_display} (unq_rate={redundant_question_sample_rate * 100:.1f}%, {redundant_question_events} total events)"
            ])
        else:
            ask_display = f"{ask_triggered_samples} / {denominator}" if denominator else "0 / 0"
            log_lines.extend([
                f"- Samples where at least one clarifying question was asked (among samples with required_points): {ask_display} (ask_rate={ask_triggered_sample_rate:.4f}, {ask_triggered_sample_rate * 100:.1f}%)",
                f"- Final answers after covering all required points: {compliant_final_answers} / {total_final_answers} ({compliance_rate * 100:.1f}%)",
                f"- Premature final answers (missing required info): {non_compliant_final_answers} ({premature_answer_rate * 100:.1f}% of final answers)",
                f"- Samples with unnecessary clarifying questions after all info was available: {redundant_question_samples} ({redundant_question_sample_rate * 100:.1f}% of valid samples, {redundant_question_events} total events)"
            ])
        log = "\n".join(log_lines) + "\n\n" + turn_distribution_log

        all_scores = [
            res.get("correct") if not res.get("skipped") else None
            for res in final_results
        ]

        return accuracy, all_scores, log
