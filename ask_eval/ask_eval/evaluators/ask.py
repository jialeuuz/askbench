import json
import os
import re
from typing import Dict, List, Tuple, Any
from argparse import Namespace
import asyncio
from collections import Counter

from tqdm import tqdm

from ask_eval.evaluators.base_evaluator import BaseEvaluator

# --- 在这里配置首轮对话的引导模式 ---
# 可选值: 'none', 'weak', 'strong'
# 'none': 不添加任何引导
# 'weak': 添加弱引导，暗示模型可以提问
# 'strong': 添加强引导，明确要求模型必须先提问
GUIDANCE_MODE = 'none'
MAX_JUDGE_JSON_RETRIES = 10

# Weak guidance: Politely suggests that the model can ask for more information.
WEAK_GUIDANCE_PROMPT = "If you need more information to provide a better and more complete answer, please feel free to ask me any questions."
# Strong guidance: A direct instruction that the model MUST ask questions before answering.
STRONG_GUIDANCE_PROMPT = "Important: My question is likely incomplete. You must ask clarifying questions to gather all the necessary information before you provide a final answer."


ARBITER_EVALUATOR_PROMPT_TEMPLATE = """
You are an expert evaluator reviewing the last response from an AI assistant that should ask for missing information before answering degraded questions.

Your responsibilities for this turn:
1. Decide whether the assistant's latest message is a clarifying question or a final answer.
2. Track whether every **required clarification point** has already been resolved in the conversation. A point counts as resolved only if the user has explicitly provided that information.
3. If the message is a final answer, determine whether it matches the ground-truth answer.

---
### Reference Information
- **Ground truth final answer:** <ground_truth_answer>
- **Original (complete) question:** <ori_question>
- **Degraded question seen by the assistant:** <degraded_question>
- **Why information is missing:** <degraded_info>

### Required clarification points (the assistant must gather all of them before giving a final answer):
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

**Clarifications you DO know but must only reveal if the assistant explicitly asks for them:**
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


def parse_json_to_dict(json_string: str) -> dict:
    """从模型的Markdown格式响应中解析出JSON字典。"""
    match = re.search(r"```json\s*([\s\S]+?)\s*```", json_string)
    if match:
        json_cleaned = match.group(1).strip()
    else:
        # 如果没有找到 markdown block，尝试直接解析整个字符串
        json_cleaned = json_string.strip()
        
    try:
        return json.loads(json_cleaned)
    except (json.JSONDecodeError, TypeError):
        return {} # 返回空字典表示解析失败


def format_required_points(points: List[str]) -> str:
    if not points:
        return "- None (the question already includes all necessary details)."
    formatted = []
    for idx, point in enumerate(points, start=1):
        formatted.append(f"{idx}. {point}")
    return "\n".join(formatted)

class AskEvaluator(BaseEvaluator):
    def __init__(self, model, eval_config: Dict, judge_model=None):
        super().__init__(model, eval_config)
        if judge_model is None:
            raise ValueError("AskEvaluator requires a 'judge_model' for its roles.")
        self.judge_model = judge_model

    async def _call_judge_with_retry(self, prompt: str) -> Dict[str, Any]:
        """Call the judge model and retry parsing JSON up to MAX_JUDGE_JSON_RETRIES times."""
        last_raw_response = ""
        for attempt in range(1, MAX_JUDGE_JSON_RETRIES + 1):
            try:
                judge_response_raw = await self.judge_model.infer_async(
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

    async def evaluate_multi_turn(self, args: Namespace, test_data: List[Dict], max_turns: int) -> Tuple[float, List[bool], str]:
        print(f"Starting turn-by-turn evaluation for {len(test_data)} samples with max {max_turns} turns...")

        forced_guidance = ''
        
        active_samples = []
        for i, sample_data in enumerate(test_data):
            initial_history = [{"role": "user", "content": sample_data["degraded_question"] + forced_guidance}]
            required_points = sample_data.get("required_points") or []
            active_samples.append({
                "id": sample_data.get("id", i),
                "data": sample_data,
                "conversation_history": initial_history,
                "turn_logs": [],
                "is_finished": False,
                "result": None,
                "required_points": required_points,
                "metrics": {
                    "redundant_question_events": 0,
                    "premature_final_answer": False
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

                llm_coroutines.append(self.model.infer_async(
                    message=messages_for_llm,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ))
            
            llm_responses_raw = await run_tasks_with_progress(llm_coroutines, f"Turn {turn}: LLM Inference")
            llm_responses = [res[0] for res in llm_responses_raw]

            for i, sample_state in enumerate(active_samples):
                sample_state["conversation_history"].append({"role": "assistant", "content": llm_responses[i]})

            # 2. Judge模型执行【仲裁+评估】
            judge_coroutines = []
            for sample_state in active_samples:
                convo_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]])
                required_points_text = format_required_points(sample_state["required_points"])
                judge_prompt_str = ARBITER_EVALUATOR_PROMPT_TEMPLATE \
                    .replace("<ground_truth_answer>", sample_state["data"]["expected_answer"]) \
                    .replace("<conversation_history>", convo_str) \
                    .replace("<ori_question>", sample_state["data"].get("ori_question", "")) \
                    .replace("<degraded_question>", sample_state["data"].get("degraded_question", "")) \
                    .replace("<degraded_info>", sample_state["data"].get("degraded_info", "")) \
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

                if not decision.get("is_final_answer") and all_points_resolved:
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
                    "information_i_have": sample_state["data"].get("degraded_info", ""),
                    "required_points_to_unlock_answer": sample_state["required_points"]
                }
                user_knowledge_str = json.dumps(user_knowledge, indent=2, ensure_ascii=False)
                convo_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]])
                assistant_question = sample_state["conversation_history"][-1]["content"]
                required_points_text = format_required_points(sample_state["required_points"])

                simulator_prompt_str = SIMULATOR_PROMPT_TEMPLATE.replace("<user_internal_knowledge>", user_knowledge_str) \
                                                                .replace("<conversation_history>", convo_str) \
                                                                .replace("<assistant_question>", assistant_question) \
                                                                .replace("<required_points>", required_points_text)
                simulator_coroutines.append(self.judge_model.infer_async(message=[{"role": "user", "content": simulator_prompt_str}], temperature=0.5))

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

        redundant_question_events = sum(res.get("behavior_metrics", {}).get("redundant_question_events", 0) for res in valid_results)
        redundant_question_samples = sum(1 for res in valid_results if res.get("behavior_metrics", {}).get("redundant_question_events", 0) > 0)
        redundant_question_sample_rate = (redundant_question_samples / denominator) if denominator else 0.0

        reason_counts = Counter(res.get("reason") for res in final_results)
        turn_distribution_log = "Evaluation Outcome Distribution:\n"
        for reason, count in reason_counts.most_common():
            percentage = (count / total_samples) * 100 if total_samples else 0.0
            turn_distribution_log += f"  - {reason}: {count} samples ({percentage:.1f}%)\n"

        log_lines = [
            f"AskMind Final Accuracy: {accuracy:.4f} ({correct_count} / {denominator})",
            f"- Valid samples: {denominator} / {total_samples} (skipped {skipped_samples})",
            f"- Final answers after covering all required points: {compliant_final_answers} / {total_final_answers} ({compliance_rate * 100:.1f}%)",
            f"- Premature final answers (missing required info): {non_compliant_final_answers} ({premature_answer_rate * 100:.1f}% of final answers)",
            f"- Samples with unnecessary clarifying questions after all info was available: {redundant_question_samples} ({redundant_question_sample_rate * 100:.1f}% of valid samples, {redundant_question_events} total events)"
        ]
        log = "\n".join(log_lines) + "\n\n" + turn_distribution_log

        all_scores = [
            res.get("correct") if not res.get("skipped") else None
            for res in final_results
        ]

        return accuracy, all_scores, log
