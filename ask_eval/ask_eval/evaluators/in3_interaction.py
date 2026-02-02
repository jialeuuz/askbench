import asyncio
import json
import os
from argparse import Namespace
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ask_eval.evaluators.ask import (
    AskEvaluator,
    FORCE_FINAL_ANSWER_PROMPT,
    GUIDANCE_MODE,
    STRONG_GUIDANCE_PROMPT,
    WEAK_GUIDANCE_PROMPT,
    format_required_points,
)

IN3_ARBITER_PROMPT_TEMPLATE = """
You are an expert evaluator reviewing the latest response from an AI assistant that must decide whether to clarify or to give a final answer for a potentially vague user task.

Context:
- Original task from the user: <ori_question>
- Flagged as vague: <vague_flag>
- Scenario background describing possible gaps: <scenario_context>

Checklist of clarifications that should be covered before the assistant finalizes the answer:
<required_points>

Conversation history (most recent message last):
<conversation_history>

Instructions:
1. Determine whether the assistant's last message is a clarifying question (continue the dialogue) or a final response.
2. Track which checklist items have been explicitly answered by the user. Mark `all_required_points_resolved` only when every checklist item has already been satisfied (or when no checklist items were provided).
3. Identify any remaining checklist items in `missing_required_points`. Use the descriptions verbatim from the list above.
4. Since this benchmark has no ground-truth final answer, always output `"is_correct": null`.

Output format:
Provide a `Reasoning:` line followed by a ```json code block:

```json
{
  "is_final_answer": boolean,
  "is_correct": null,
  "all_required_points_resolved": boolean,
  "missing_required_points": ["..."],
  "notes": "optional short rationale"
}
```
""".strip()


IN3_SIMULATOR_PROMPT_TEMPLATE = """
You are acting as the human user. Reply naturally based on the private profile below and only reveal information that the assistant explicitly asks for.

Your hidden profile:
```json
<user_profile>
```

Guidelines:
- Answer only the specific clarifying question being asked. Keep replies short and human-like.
- When a question matches one of the keys inside `clarification_answers`, respond with the corresponding value.
- If the question is unrelated or repeats information you've already provided, politely mention that you've already shared everything relevant.
- If no clarification is needed (the task is not vague), gently remind the assistant that the request already contains all necessary details and ask them to proceed with the solution.

Conversation history so far:
<conversation_history>

Assistant's latest question:
"<assistant_question>"

Only return the text of your reply.
""".strip()


class In3InteractionEvaluator(AskEvaluator):
    """Judge-driven evaluator dedicated to the in3_interaction benchmark."""

    def __init__(self, model, eval_config: Dict, judge_model=None, judge_config: Dict = None):
        super().__init__(model, eval_config, judge_model=judge_model, judge_config=judge_config)
        self.task_label = eval_config.get("task_label", "in3_interaction")

    def _normalize_sample(self, sample_data: Dict[str, Any], fallback_id: int) -> Dict[str, Any]:
        """Map dataset fields into the standard structure required by the evaluator."""
        ori_question = (
            sample_data.get("ori_question")
            or sample_data.get("task")
            or sample_data.get("question")
        )
        if not ori_question:
            raise ValueError("in3_interaction sample is missing the 'task' or 'ori_question' field.")

        degraded_question = sample_data.get("degraded_question") or ori_question

        def _stringify_info(value: Any) -> str:
            if not value:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                return "\n".join(parts)
            return json.dumps(value, ensure_ascii=False)

        required_points = sample_data.get("required_points")
        required_point_answers = sample_data.get("required_point_answers")
        degraded_info = _stringify_info(sample_data.get("degraded_info"))

        if required_points is None:
            # Backward compatibility: derive required_points from missing_details if needed.
            missing_details: List[Dict[str, Any]] = sample_data.get("missing_details") or []
            derived_points: List[str] = []
            detail_info_lines: List[str] = []
            detail_answers: Dict[str, str] = {}

            user_responses = [
                action.get("content", "")
                for action in sample_data.get("actions", [])
                if action.get("role") == "user"
            ]
            user_resp_idx = 0

            for detail_idx, detail in enumerate(missing_details, start=1):
                description = detail.get("description") or detail.get("inquiry") or f"Detail {detail_idx}"
                derived_points.append(description)

                inquiry = detail.get("inquiry") or ""
                importance = detail.get("importance") or "?"
                options = detail.get("options") or []
                option_hint = f" (Options: {', '.join(options)})" if options else ""
                detail_info_lines.append(f"- {description} (importance {importance}): {inquiry}{option_hint}")

                answer_text: Optional[str] = None
                if user_resp_idx < len(user_responses):
                    answer_text = user_responses[user_resp_idx].strip()
                    user_resp_idx += 1
                if not answer_text:
                    if options:
                        answer_text = f"I'd go with {options[0]}."
                    else:
                        answer_text = "I trust your judgement—feel free to suggest what fits best."
                detail_answers[description] = answer_text

            required_points = derived_points
            required_point_answers = detail_answers
            if not degraded_info:
                degraded_info = "\n".join(detail_info_lines) if detail_info_lines else "No additional clarifications are expected for this task."
        else:
            required_points = [str(point) for point in required_points]
            required_point_answers = required_point_answers or {}
            if not degraded_info:
                degraded_info = "No additional clarifications are expected for this task."

        normalized = {
            "id": sample_data.get("id", fallback_id),
            "ori_question": ori_question,
            "degraded_question": degraded_question,
            "required_points": required_points,
            "required_point_answers": required_point_answers,
            "degraded_info": degraded_info,
            "vague": bool(sample_data.get("vague", False)),
            "category": sample_data.get("category", ""),
        }
        return normalized

    def _build_scenario_fields(self, normalized_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Construct the scenario metadata consumed by the judge prompts."""
        return {
            "type": "in3_interaction",
            "question_text": normalized_sample["ori_question"],
            "info_text": normalized_sample.get("degraded_info", ""),
            "points": list(normalized_sample.get("required_points", [])),
            "question_header": "Original task shown to the assistant",
            "info_header": "Context about potential missing information",
            "points_header": "Clarifications required before finalizing the response",
        }

    async def evaluate_multi_turn(
        self,
        args: Namespace,
        test_data: List[Dict],
        max_turns: int,
    ) -> Tuple[Optional[float], List[Optional[bool]], str]:
        """Run the multi-turn loop without computing accuracy (no ground truth answers)."""
        print(f"Starting in3_interaction evaluation for {len(test_data)} samples with max {max_turns} turns...")
        self._ensure_semaphores()

        active_samples = []
        for i, raw_sample in enumerate(test_data):
            normalized = self._normalize_sample(raw_sample, i)
            scenario_fields = self._build_scenario_fields(normalized)
            initial_prompt = self._build_initial_prompt(normalized, scenario_fields)
            initial_history = [{"role": "user", "content": initial_prompt}]
            required_points = list(scenario_fields["points"])
            sample_state = {
                "id": normalized["id"],
                "data": normalized,
                "conversation_history": initial_history,
                "turn_logs": [],
                "is_finished": False,
                "result": None,
                "required_points": required_points,
                "scenario": scenario_fields,
                "required_point_answers": normalized.get("required_point_answers", {}),
                "metrics": {
                    "redundant_question_events": 0,
                    "premature_final_answer": False,
                    "asked_any_question": False,
                },
                "last_known_missing_points": list(required_points),
                "last_all_required_points_resolved": False if required_points else True,
                "skipped": False,
            }
            active_samples.append(sample_state)

        final_results = []

        for turn in range(1, max_turns + 1):
            if not active_samples:
                print("All samples have been evaluated. Finishing early.")
                break

            print(f"\n===== Turn {turn}/{max_turns} | Active Samples: {len(active_samples)} =====")

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

            llm_coroutines = []
            for sample_state in active_samples:
                messages_for_llm = list(sample_state["conversation_history"])

                if turn == 1 and GUIDANCE_MODE != "none" and messages_for_llm:
                    prompt_to_add = ""
                    if GUIDANCE_MODE == "weak":
                        prompt_to_add = WEAK_GUIDANCE_PROMPT
                    elif GUIDANCE_MODE == "strong":
                        prompt_to_add = STRONG_GUIDANCE_PROMPT

                    if prompt_to_add:
                        last_message = messages_for_llm[-1].copy()
                        last_message["content"] += f"\n\n{prompt_to_add}"
                        messages_for_llm[-1] = last_message

                if turn == max_turns and messages_for_llm:
                    last_message = messages_for_llm[-1].copy()
                    last_message["content"] += "\n" + FORCE_FINAL_ANSWER_PROMPT
                    messages_for_llm[-1] = last_message

                llm_coroutines.append(self._model_infer(messages_for_llm))

            llm_responses_raw = await run_tasks_with_progress(llm_coroutines, f"Turn {turn}: LLM Inference")
            llm_responses = [res[0] for res in llm_responses_raw]

            for idx, sample_state in enumerate(active_samples):
                sample_state["conversation_history"].append(
                    {"role": "assistant", "content": llm_responses[idx]}
                )

            judge_coroutines = []
            for sample_state in active_samples:
                convo_str = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]]
                )
                required_points_text = format_required_points(sample_state["required_points"])
                scenario_info = sample_state["scenario"]["info_text"] or "None provided."
                scenario_question = sample_state["scenario"]["question_text"]
                vague_flag = "true" if sample_state["data"].get("vague") else "false"
                judge_prompt_str = (
                    IN3_ARBITER_PROMPT_TEMPLATE.replace("<ori_question>", sample_state["data"]["ori_question"])
                    .replace("<scenario_question>", scenario_question)
                    .replace("<scenario_context>", scenario_info)
                    .replace("<required_points>", required_points_text)
                    .replace("<conversation_history>", convo_str)
                    .replace("<vague_flag>", vague_flag)
                )
                judge_coroutines.append(self._call_judge_with_retry(judge_prompt_str))

            judge_results = await run_tasks_with_progress(judge_coroutines, f"Turn {turn}: Judging")

            still_active_samples = []
            for idx, sample_state in enumerate(active_samples):
                judge_result = judge_results[idx]
                decision = judge_result.get("decision")
                turn_log = {
                    "turn": turn,
                    "conversation_at_turn": json.loads(json.dumps(sample_state["conversation_history"])),
                    "judge_decision": decision,
                    "judge_raw_response": judge_result.get("raw_response"),
                    "judge_parse_attempts": judge_result.get("attempts"),
                    "judge_parse_success": judge_result.get("success", False),
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
                        "ground_truth_answer": None,
                        "turn_logs": sample_state["turn_logs"],
                        "is_final_answer": False,
                        "missing_required_points": sample_state["last_known_missing_points"],
                        "all_required_points_resolved": sample_state["last_all_required_points_resolved"],
                        "required_points": sample_state["required_points"],
                        "skipped": True,
                        "behavior_metrics": dict(sample_state["metrics"]),
                        "metadata": {"vague": sample_state["data"].get("vague", False)},
                    }
                    final_results.append(sample_state["result"])
                    continue

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

                if not decision.get("is_final_answer"):
                    sample_state["metrics"]["asked_any_question"] = True
                    if all_points_resolved:
                        sample_state["metrics"]["redundant_question_events"] += 1

                if decision.get("is_final_answer"):
                    sample_state["is_finished"] = True
                    result_reason = (
                        "FinalAnswerAfterClarifications" if all_points_resolved else "FinalAnswerMissingInfo"
                    )
                    sample_state["metrics"]["premature_final_answer"] = not all_points_resolved
                    sample_state["result"] = {
                        "id": sample_state["id"],
                        "correct": None,
                        "reason": result_reason,
                        "final_turn": turn,
                        "conversation_history": sample_state["conversation_history"],
                        "ground_truth_answer": None,
                        "turn_logs": sample_state["turn_logs"],
                        "is_final_answer": True,
                        "missing_required_points": missing_points,
                        "all_required_points_resolved": all_points_resolved,
                        "required_points": sample_state["required_points"],
                        "skipped": False,
                        "behavior_metrics": dict(sample_state["metrics"]),
                        "metadata": {"vague": sample_state["data"].get("vague", False)},
                    }
                    final_results.append(sample_state["result"])
                else:
                    if turn < max_turns:
                        still_active_samples.append(sample_state)
                    else:
                        sample_state["is_finished"] = True
                        sample_state["result"] = {
                            "id": sample_state["id"],
                            "correct": None,
                            "reason": "FailedToAnswerOnLastTurn",
                            "final_turn": max_turns,
                            "conversation_history": sample_state["conversation_history"],
                            "ground_truth_answer": None,
                            "turn_logs": sample_state["turn_logs"],
                            "is_final_answer": False,
                            "missing_required_points": sample_state["last_known_missing_points"],
                            "all_required_points_resolved": sample_state["last_all_required_points_resolved"],
                            "required_points": sample_state["required_points"],
                            "skipped": False,
                            "behavior_metrics": dict(sample_state["metrics"]),
                            "metadata": {"vague": sample_state["data"].get("vague", False)},
                        }
                        final_results.append(sample_state["result"])

            active_samples = still_active_samples
            if not active_samples:
                continue

            simulator_coroutines = []
            for sample_state in active_samples:
                user_profile = {
                    "original_task": sample_state["data"]["ori_question"],
                    "vague": sample_state["data"].get("vague", False),
                    "clarification_answers": sample_state.get("required_point_answers", {}),
                }
                convo_str = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]]
                )
                assistant_question = sample_state["conversation_history"][-1]["content"]
                user_profile_str = json.dumps(user_profile, indent=2, ensure_ascii=False)

                simulator_prompt_str = (
                    IN3_SIMULATOR_PROMPT_TEMPLATE.replace("<user_profile>", user_profile_str)
                    .replace("<conversation_history>", convo_str)
                    .replace("<assistant_question>", assistant_question)
                )
                simulator_coroutines.append(
                    self._judge_infer(message=[{"role": "user", "content": simulator_prompt_str}], temperature=0.3)
                )

            simulated_responses_raw = await run_tasks_with_progress(
                simulator_coroutines, f"Turn {turn}: Simulating User"
            )
            simulated_responses = [res[0] for res in simulated_responses_raw]

            for idx, sample_state in enumerate(active_samples):
                next_user_query = simulated_responses[idx]
                sample_state["turn_logs"][-1]["simulated_user_response"] = next_user_query
                sample_state["conversation_history"].append({"role": "user", "content": next_user_query})

        for sample_state in active_samples:
            if not sample_state["is_finished"]:
                sample_state["is_finished"] = True
                sample_state["result"] = {
                    "id": sample_state["id"],
                    "correct": None,
                    "reason": "MaxTurnsReached",
                    "final_turn": max_turns,
                    "conversation_history": sample_state["conversation_history"],
                    "ground_truth_answer": None,
                    "turn_logs": sample_state["turn_logs"],
                    "is_final_answer": False,
                    "missing_required_points": sample_state["last_known_missing_points"],
                    "all_required_points_resolved": sample_state["last_all_required_points_resolved"],
                    "required_points": sample_state["required_points"],
                    "skipped": False,
                    "behavior_metrics": dict(sample_state["metrics"]),
                    "metadata": {"vague": sample_state["data"].get("vague", False)},
                }
                final_results.append(sample_state["result"])

        output_file = os.path.join(args.save_dir, "askbench_detailed_results.json")
        os.makedirs(args.save_dir, exist_ok=True)
        final_results.sort(key=lambda x: x["id"])
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed evaluation logs saved to: {output_file}")

        total_samples = len(final_results)
        valid_results = [res for res in final_results if not res.get("skipped")]
        skipped_samples = total_samples - len(valid_results)

        final_answers = [res for res in valid_results if res.get("is_final_answer")]
        total_final_answers = len(final_answers)
        compliant_final_answers = sum(1 for res in final_answers if res.get("all_required_points_resolved"))
        non_compliant_final_answers = total_final_answers - compliant_final_answers
        compliance_rate = (compliant_final_answers / total_final_answers) if total_final_answers else 0.0
        premature_answer_rate = (non_compliant_final_answers / total_final_answers) if total_final_answers else 0.0

        redundant_question_events = sum(
            res.get("behavior_metrics", {}).get("redundant_question_events", 0) for res in valid_results
        )
        redundant_question_samples = sum(
            1
            for res in valid_results
            if res.get("behavior_metrics", {}).get("redundant_question_events", 0) > 0
        )
        redundant_question_sample_rate = (
            (redundant_question_samples / len(valid_results)) if valid_results else 0.0
        )

        ask_triggered_samples = sum(
            1 for res in valid_results if res.get("behavior_metrics", {}).get("asked_any_question")
        )
        ask_triggered_sample_rate = (ask_triggered_samples / len(valid_results)) if valid_results else 0.0

        vague_results = [res for res in valid_results if res.get("metadata", {}).get("vague")]
        clear_results = [res for res in valid_results if not res.get("metadata", {}).get("vague")]
        vague_triggered = sum(
            1 for res in vague_results if res.get("behavior_metrics", {}).get("asked_any_question")
        )
        clear_without_questions = sum(
            1 for res in clear_results if not res.get("behavior_metrics", {}).get("asked_any_question")
        )
        vague_ask_rate = (vague_triggered / len(vague_results)) if vague_results else 0.0
        clear_direct_rate = (clear_without_questions / len(clear_results)) if clear_results else 0.0

        reason_counts = Counter(res.get("reason") for res in final_results)
        turn_distribution_log = "Evaluation Outcome Distribution:\n"
        for reason, count in reason_counts.most_common():
            percentage = (count / total_samples) * 100 if total_samples else 0.0
            turn_distribution_log += f"  - {reason}: {count} samples ({percentage:.1f}%)\n"

        display_label = getattr(self, "task_label", "in3_interaction")
        log_lines = [
            f"{display_label} Vague Ask Rate: {vague_ask_rate:.4f} ({vague_triggered} / {len(vague_results) or 0})",
            f"- Clear-task direct answer rate: {clear_direct_rate:.4f} ({clear_without_questions} / {len(clear_results) or 0})",
            f"- Valid samples: {len(valid_results)} / {total_samples} (skipped {skipped_samples})",
            f"- Clarifying question coverage (all required points addressed before answering): {compliant_final_answers} / {total_final_answers} ({compliance_rate * 100:.1f}%)",
            f"- Premature final answers (missing clarifications): {non_compliant_final_answers} ({premature_answer_rate * 100:.1f}% of final answers)",
            f"- Samples where at least one clarifying question was asked: {ask_triggered_samples} / {len(valid_results) or 0} (ask_rate={ask_triggered_sample_rate:.4f}, {ask_triggered_sample_rate * 100:.1f}%)",
            f"- Samples with redundant questions after all info was available: {redundant_question_samples} / {len(valid_results) or 0} (unq_rate={redundant_question_sample_rate:.4f}, {redundant_question_events} total events)",
        ]
        log = "\n".join(log_lines) + "\n\n" + turn_distribution_log

        all_scores = [None for _ in final_results]
        return None, all_scores, log
