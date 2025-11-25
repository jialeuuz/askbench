import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ask_eval.evaluators.judge_utils import MAX_JUDGE_JSON_RETRIES, parse_json_to_dict


class JudgeEvaluatorMixin:
    """Mixin providing judge-model based evaluation for single-turn tasks."""

    requires_judge = True

    def __init__(self, *args, judge_model=None, judge_config=None, **kwargs):
        if judge_model is None:
            raise ValueError("JudgeEvaluatorMixin requires a judge_model instance.")
        self.judge_model = judge_model
        self.judge_config = judge_config or {}
        self._judge_temperature = float(self.judge_config.get("temperature", 0.0))
        self._judge_max_tokens = int(self.judge_config.get("max_new_tokens", 2048))
        max_concurrent = int(self.judge_config.get("max_concurrent", 10))
        self._judge_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        super().__init__(*args, **kwargs)

    async def evaluate_responses_async(
        self,
        args,
        test_data: List[Dict[str, Any]],
        responses: List[str],
        thinking_processes: List[str],
        truncated_flags: List[str],
        prompts: List[str],
    ) -> tuple:
        """Evaluate responses via the judge model with retry + skip support."""
        extracted_answers = [self.extract_answer(resp) for resp in responses]
        judge_prompts = [
            self.build_judge_prompt(sample, prompt, resp, extracted)
            for sample, prompt, resp, extracted in zip(test_data, prompts, responses, extracted_answers)
        ]

        judge_coroutines = [self._call_judge_with_retry(prompt) for prompt in judge_prompts]
        judge_results = await self._gather_with_progress(judge_coroutines, desc="Judging")

        response_records = []
        cors: List[Optional[int]] = []
        truncation_stats = {"not_truncated": 0, "truncated": 0, "none": 0}
        valid_count = 0
        correct_count = 0
        skipped = 0

        for idx, (sample, response, extracted, thinking, truncated, prompt) in enumerate(
            zip(test_data, responses, extracted_answers, thinking_processes, truncated_flags, prompts)
        ):
            truncation_stats[truncated] = truncation_stats.get(truncated, 0) + 1
            judge_payload = judge_results[idx]
            record = {
                "question": prompt,
                "response": response,
                "extracted_answer": extracted,
                "reference_answer": self.reference_answer_for_record(sample),
                "thinking_process": thinking,
                "truncated": truncated,
                "judge": {
                    "decision": judge_payload.get("decision"),
                    "raw_response": judge_payload.get("raw_response"),
                    "parse_attempts": judge_payload.get("attempts"),
                    "parse_success": judge_payload.get("success", False),
                },
            }

            if not judge_payload.get("success"):
                cors.append(None)
                skipped += 1
                record["correct"] = None
                record["skipped"] = True
                record["skip_reason"] = "JudgeJSONParseFailed"
            else:
                decision = self.interpret_judge_decision(judge_payload.get("decision"))
                if decision is None:
                    cors.append(None)
                    skipped += 1
                    record["correct"] = None
                    record["skipped"] = True
                    record["skip_reason"] = "JudgeResultInvalid"
                else:
                    cors.append(decision)
                    record["correct"] = decision
                    record["skipped"] = False
                    valid_count += 1
                    correct_count += decision

            response_records.append(record)

        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "api_responses.json"), "w", encoding="utf-8") as f:
            json.dump(response_records, f, indent=2, ensure_ascii=False)

        accuracy = correct_count / valid_count if valid_count else 0.0
        log = (
            f"Judge accuracy (valid {valid_count}/{len(test_data)} samples): {accuracy:.3f}\n"
            f"Skipped samples: {skipped}\n"
            "Truncation statistics:\n"
        )
        for status, count in truncation_stats.items():
            percentage = count / len(responses) * 100 if responses else 0
            log += f"- {status}: {count} ({percentage:.1f}%)\n"

        return accuracy, cors, log

    def reference_answer_for_record(self, sample: Dict[str, Any]) -> Any:
        """Allow subclasses to customise how reference answers are logged."""
        return sample.get("expected_answer")

    async def _call_judge_with_retry(self, prompt: str) -> Dict[str, Any]:
        last_raw_response = ""
        for attempt in range(1, MAX_JUDGE_JSON_RETRIES + 1):
            try:
                async with self._judge_semaphore:
                    judge_response = await self.judge_model.infer_async(
                        message=[{"role": "user", "content": prompt}],
                        temperature=self._judge_temperature,
                        max_tokens=self._judge_max_tokens,
                    )
                last_raw_response = judge_response[0]
                parsed = parse_json_to_dict(last_raw_response)
                if parsed:
                    return {
                        "success": True,
                        "decision": parsed,
                        "raw_response": last_raw_response,
                        "attempts": attempt,
                    }
            except Exception as exc:
                last_raw_response = f"Error invoking judge: {exc}"

        return {
            "success": False,
            "decision": None,
            "raw_response": last_raw_response,
            "attempts": MAX_JUDGE_JSON_RETRIES,
        }

    async def _gather_with_progress(self, coroutines: List[asyncio.Future], desc: str):
        if not coroutines:
            return []
        tasks = []
        with tqdm(total=len(coroutines), desc=desc) as pbar:
            def _update(_):
                pbar.update(1)
            for coro in coroutines:
                task = asyncio.create_task(coro)
                task.add_done_callback(_update)
                tasks.append(task)
            results = await asyncio.gather(*tasks)
        return results

    def interpret_judge_decision(self, decision: Optional[Dict[str, Any]]) -> Optional[int]:
        if not isinstance(decision, dict):
            return None
        verdict = decision.get("result")
        if isinstance(verdict, bool):
            return 1 if verdict else 0
        if isinstance(verdict, str):
            verdict_normalized = verdict.strip().lower()
            if verdict_normalized in {"correct", "right", "yes", "match"}:
                return 1
            if verdict_normalized in {"incorrect", "wrong", "no"}:
                return 0
        return None

    def build_judge_prompt(
        self, sample: Dict[str, Any], prompt: str, model_response: str, extracted_answer: str
    ) -> str:
        """Subclasses must provide the full prompt sent to the judge."""
        raise NotImplementedError
