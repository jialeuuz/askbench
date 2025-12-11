import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ask_eval.evaluators.base_evaluator import BaseEvaluator
from ask_eval.evaluators.judge_utils import MAX_JUDGE_JSON_RETRIES, parse_json_to_dict


class HealthBenchEvaluator(BaseEvaluator):
    """单轮对话 + rubric 打分的 HealthBench 评估器。"""

    requires_judge = True
    metric_label = "HealthBench Score"

    def __init__(self, model, eval_config: Dict, judge_model=None, judge_config: Dict = None):
        if judge_model is None:
            raise ValueError("HealthBenchEvaluator requires a judge/grader model.")
        super().__init__(model, eval_config)
        self.judge_model = judge_model
        self.judge_config = judge_config or {}
        self.grader_temperature = float(self.judge_config.get("temperature", 0.0))
        self.grader_max_tokens = int(self.judge_config.get("max_new_tokens", 2048))
        self._grader_semaphore = asyncio.Semaphore(int(self.judge_config.get("max_concurrent", 10)))
        self.grader_template = self._load_grader_template()

    def _load_grader_template(self) -> str:
        """加载 data/common/healthbench/grader_prompt.py 中的模板。"""
        template_path = Path(__file__).resolve().parents[2] / "data" / "common" / "healthbench" / "grader_prompt.py"
        if not template_path.exists():
            return ""
        namespace: Dict[str, Any] = {}
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                code = f.read()
            exec(code, namespace)
            template = namespace.get("GRADER_TEMPLATE", "")
            return template if isinstance(template, str) else ""
        except Exception:
            return ""

    async def infer_batch(
        self,
        test_data: List[Dict],
        train_data: List[Dict] = None
    ) -> Tuple[List[str], List[str], List[str], List[Any]]:
        """直接使用样本内的 prompt 作为对话历史调用模型。"""
        prompts = [sample.get("prompt", []) for sample in test_data]
        try:
            responses, thinking_processes, truncated_flags = await self.model.infer_batch_async(
                prompts, self.max_tokens, self.temperature, self.max_concurrent
            )
        except Exception as exc:
            print(f"Error generating response: {exc}")
            responses = ["Error"] * len(prompts)
            thinking_processes = ["none"] * len(prompts)
            truncated_flags = ["none"] * len(prompts)
        return responses, thinking_processes, truncated_flags, prompts

    async def _call_grader_with_retry(self, prompt: str) -> Dict[str, Any]:
        last_raw_response = ""
        for attempt in range(1, MAX_JUDGE_JSON_RETRIES + 1):
            try:
                async with self._grader_semaphore:
                    grader_response = await self.judge_model.infer_async(
                        message=[{"role": "user", "content": prompt}],
                        temperature=self.grader_temperature,
                        max_tokens=self.grader_max_tokens,
                    )
                last_raw_response = grader_response[0]
                parsed = parse_json_to_dict(last_raw_response)
                if parsed:
                    return {
                        "success": True,
                        "decision": parsed,
                        "raw_response": last_raw_response,
                        "attempts": attempt,
                    }
            except Exception as exc:
                last_raw_response = f"Error invoking grader: {exc}"
        return {
            "success": False,
            "decision": None,
            "raw_response": last_raw_response,
            "attempts": MAX_JUDGE_JSON_RETRIES,
        }

    def _format_conversation(self, prompt_messages: Any, assistant_reply: str) -> Tuple[str, List[Dict[str, str]]]:
        """将对话历史与模型回复串成 plain text，供 grader 使用。"""
        history: List[Dict[str, str]] = []
        if isinstance(prompt_messages, list):
            history.extend(prompt_messages)
        else:
            history.append({"role": "user", "content": str(prompt_messages)})
        history.append({"role": "assistant", "content": assistant_reply})
        convo_lines = [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in history]
        return "\n".join(convo_lines), history

    def _build_grader_prompt(self, conversation_text: str, rubric_text: str) -> str:
        rubric_text = rubric_text or ""
        if self.grader_template:
            return (
                self.grader_template
                .replace("<<conversation>>", conversation_text)
                .replace("<<rubric_item>>", rubric_text)
            )
        return (
            "Conversation:\n"
            f"{conversation_text}\n\n"
            "Rubric item:\n"
            f"{rubric_text}\n\n"
            'Return a JSON markdown block {"explanation": "...", "criteria_met": true/false}.'
        )

    @staticmethod
    def _normalize_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        return None

    async def evaluate_responses_async(
        self,
        args,
        test_data: List[Dict[str, Any]],
        responses: List[str],
        thinking_processes: List[str],
        truncated_flags: List[str],
        prompts: List[Any],
    ) -> tuple:
        records = []
        per_sample_scores: List[Optional[float]] = []
        grader_failures = 0
        total_rubrics = 0

        for idx, (sample, model_resp, thinking, truncated) in enumerate(
            zip(test_data, responses, thinking_processes, truncated_flags)
        ):
            prompt_messages = prompts[idx] if idx < len(prompts) else sample.get("prompt", [])
            conversation_text, conversation_history = self._format_conversation(prompt_messages, model_resp)
            rubrics = sample.get("rubrics", []) or []
            total_rubrics += len(rubrics)
            rubric_prompts = [
                self._build_grader_prompt(conversation_text, str(rubric.get("criterion", "")))
                for rubric in rubrics
            ]

            grader_tasks = [self._call_grader_with_retry(p) for p in rubric_prompts]
            grader_results = await asyncio.gather(*grader_tasks) if grader_tasks else []

            raw_score = 0.0
            valid_rubric_count = 0
            rubric_logs = []

            for rubric, grader_output in zip(rubrics, grader_results):
                rubric_points = float(rubric.get("points", 0))
                entry = {
                    "criterion": rubric.get("criterion", ""),
                    "points": rubric_points,
                    "grader_raw_response": grader_output.get("raw_response"),
                    "grader_parse_attempts": grader_output.get("attempts"),
                    "grader_parse_success": grader_output.get("success", False),
                }
                if grader_output.get("success"):
                    decision = grader_output.get("decision") or {}
                    criteria_met = self._normalize_bool(decision.get("criteria_met"))
                    entry["criteria_met"] = criteria_met
                    entry["explanation"] = decision.get("explanation")
                    if criteria_met is not None:
                        valid_rubric_count += 1
                        if criteria_met:
                            raw_score += rubric_points
                else:
                    grader_failures += 1
                rubric_logs.append(entry)

            positive_total = sum(max(float(r.get("points", 0)), 0.0) for r in rubrics)
            abs_total = sum(abs(float(r.get("points", 0))) for r in rubrics)
            denom = positive_total if positive_total > 0 else (abs_total if abs_total > 0 else 1.0)
            normalized_score = raw_score / denom
            normalized_score = max(0.0, min(1.0, normalized_score))

            per_sample_scores.append(normalized_score if valid_rubric_count > 0 else None)

            record = {
                "id": sample.get("id", idx),
                "canary": sample.get("canary"),
                "conversation": conversation_history,
                "conversation_text": conversation_text,
                "assistant_response": model_resp,
                "thinking_process": thinking,
                "truncated": truncated,
                "rubrics": rubric_logs,
                "raw_score": raw_score,
                "positive_total": positive_total,
                "denominator_used": denom,
                "normalized_score": normalized_score,
                "valid_rubric_count": valid_rubric_count,
                "total_rubric_count": len(rubrics),
            }
            records.append(record)

        os.makedirs(args.save_dir, exist_ok=True)
        detailed_path = os.path.join(args.save_dir, "healthbench_detailed_results.json")
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        api_path = os.path.join(args.save_dir, "api_responses.json")
        with open(api_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        valid_scores = [s for s in per_sample_scores if s is not None]
        average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        log_lines = [
            f"HealthBench average normalized score: {average_score:.4f} (valid {len(valid_scores)}/{len(test_data)})",
            f"Total rubric items: {total_rubrics}, grader parse failures: {grader_failures}",
        ]
        log = "\n".join(log_lines)
        return average_score, per_sample_scores, log
