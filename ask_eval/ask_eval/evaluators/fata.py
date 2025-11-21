import asyncio
import json
import os
from typing import Any, Dict, List, Tuple
from tqdm.asyncio import tqdm

from ask_eval.evaluators.base_evaluator import BaseEvaluator
from ask_eval.evaluators.ask import parse_json_to_dict

FATA_STAGE1_PROMPT_TEMPLATE = """
You are a senior expert in this field.

The user's question is:
<degraded_question>

Before answering, produce a structured checklist of the key follow-up questions you need to ask the user so you can uncover all missing details in a single turn.
Requirements:
1) Cover aspects such as background/context, constraints, goals and preferences, environmental factors, and historical information when relevant.
2) Organize the output into clear sections or bullet groups with concise questions.
3) Whenever a question may confuse non-expert users, provide example answer options in parentheses to illustrate what you are asking for.
4) If the information is already sufficient, respond with "The information is sufficient, no follow-up questions are needed."

Do not request sensitive personal data (phone numbers, ID numbers, or exact home addresses).
Only output the clarifying-question checklist.
""".strip()

FATA_STAGE2_PROMPT_TEMPLATE = """
You are a senior expert in this field.

Original problem:
<degraded_question>

User replies to your clarifying checklist:
<qa_block>

Based on the complete context, deliver a personalized and actionable solution.
Requirements:
1) Explicitly connect your advice to the user's objectives and constraints.
2) Provide concrete steps, strategies, and cautions.
3) If critical information is still missing, first point out the gap, explain why it matters, and then offer the best recommendation possible with the available data.

Keep the tone positive, clear, and easy to read.
""".strip()

FATA_STAGE1_SIMULATOR_PROMPT = """
You simulate the real user who asked the question. Answer the assistant's clarifying questions using the information below.

Original complete question (hidden from the assistant):
<ori_question>

Degraded question shown to the assistant:
<degraded_question>

Scenario context describing what was removed or obscured:
<degraded_info>

Key checkpoints that must be provided:
<required_points>

Assistant's clarifying question checklist:
<assistant_questions>

Instructions:
- Respond naturally in English as the user.
- Provide specific facts exactly as they appear in the original question or scenario context.
- Cover every question. If a question cannot be answered with the available information, state that the information is unavailable.
- Format the reply as a numbered list where each item restates the assistant's question followed by "Answer: ...".
""".strip()

FATA_FINAL_JUDGE_PROMPT = """
You are grading whether the assistant's final response correctly solves the original problem.

Original complete question:
<ori_question>

Degraded question shown to the assistant:
<degraded_question>

Ground-truth answer:
<expected_answer>

Assistant's final response:
<assistant_answer>

Return your decision as a Markdown ```json block with the schema:
{{
  "is_correct": true or false,
  "reason": "brief justification"
}}
Only output the JSON block.
""".strip()


def _format_required_points(points: List[str]) -> str:
    if not points:
        return "- None provided."
    return "\n".join(f"- {item}" for item in points)


class FataEvaluator(BaseEvaluator):
    """Two-stage FATA evaluator: Stage F1 clarification + Stage F2 final answer."""

    def __init__(self, model, eval_config: Dict, judge_model, judge_config: Dict):
        super().__init__(model, eval_config)
        if judge_model is None:
            raise ValueError("FataEvaluator requires a judge model.")
        self.judge_model = judge_model
        self.judge_config = judge_config or {}
        self.model_max_concurrent = self._to_int(eval_config.get("max_concurrent"), 5)
        self.model_semaphore = asyncio.Semaphore(self.model_max_concurrent)
        self.judge_max_concurrent = self._to_int(self.judge_config.get("max_concurrent"), 5)
        self.judge_semaphore = asyncio.Semaphore(self.judge_max_concurrent)
        self.max_tokens = self._to_int(eval_config.get("max_tokens"), 4096)
        self.model_timeout = self._to_float(eval_config.get("timeout"), None)
        self.stage1_temperature = self._to_float(eval_config.get("fata_stage1_temperature"), self._to_float(eval_config.get("temperature"), 0.6))
        self.stage2_temperature = self._to_float(eval_config.get("fata_stage2_temperature"), self._to_float(eval_config.get("temperature"), 0.6))
        self.judge_max_tokens = self._to_int(self.judge_config.get("max_new_tokens") or self.judge_config.get("max_tokens"), 2048)
        self.simulator_temperature = self._to_float(self.judge_config.get("simulator_temperature"), 0.3)
        self.final_judge_temperature = self._to_float(self.judge_config.get("final_temperature"), self._to_float(self.judge_config.get("temperature"), 0.0))
        self.judge_timeout = self._to_float(self.judge_config.get("timeout"), None)
        self.judge_json_retries = self._to_int(self.judge_config.get("json_retries") or self.eval_config.get("json_retries"), 6)

    def _to_int(self, value: Any, default: int) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_float(self, value: Any, default: float) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default if default is not None else 0.0

    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        return data.get("degraded_question") or data.get("ori_question", "")

    def validate_answer(self, prediction: str, reference: str) -> bool:
        return prediction.strip() == reference.strip()

    async def validate_answer_async(self, prediction: str, reference: str) -> bool:
        return self.validate_answer(prediction, reference)

    async def evaluate_dataset(self, args, test_data: List[Dict]) -> Tuple[float, List[int], str, List[Dict]]:
        os.makedirs(args.save_dir, exist_ok=True)
        results: List[Dict[str, Any]] = [None] * len(test_data)

        async def process(idx: int, sample: Dict):
            results[idx] = await self._process_single_sample(sample, idx)

        await tqdm.gather(
            *(process(idx, sample) for idx, sample in enumerate(test_data)),
            desc="Processing FATA samples",
            total=len(test_data)
        )

        detailed_path = os.path.join(args.save_dir, "fata_detailed_results.json")
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        summary_records = [
            {
                "id": item["id"],
                "is_correct": item["is_correct"],
                "judge_reason": item["final_judge"].get("reason"),
                "stage1_truncated": item["stage1_truncated"],
                "stage2_truncated": item["stage2_truncated"]
            }
            for item in results
        ]
        summary_path = os.path.join(args.save_dir, "summary_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_records, f, indent=2, ensure_ascii=False)

        total = len(results)
        correct = sum(1 for item in results if item["is_correct"])
        stage1_errors = sum(1 for item in results if item.get("stage1_error"))
        simulator_errors = sum(1 for item in results if item.get("simulator_error"))
        judge_failures = sum(1 for item in results if item["final_judge"].get("_parse_error"))

        accuracy = correct / total if total else 0.0
        log_lines = [
            f"FATA samples: {total}",
            f"Accuracy: {accuracy:.4f} ({correct}/{total})",
            f"Stage-1 generation errors: {stage1_errors}",
            f"User-simulator errors: {simulator_errors}",
            f"Final judge parse failures: {judge_failures}",
            f"Detailed results: {detailed_path}"
        ]
        log = "\n".join(log_lines)
        per_sample_scores = [1 if item["is_correct"] else 0 for item in results]
        return accuracy, per_sample_scores, log, results

    async def _process_single_sample(self, sample: Dict, idx: int) -> Dict[str, Any]:
        sample_id = sample.get("id", idx)
        stage1_prompt = self._build_stage1_prompt(sample)
        stage1_response, stage1_thinking, stage1_truncated, stage1_error = await self._call_test_model(
            stage1_prompt, self.stage1_temperature
        )

        stage1_questions = stage1_response.strip() if stage1_response else ""
        stage1_simulator_prompt = None
        stage1_simulator_reply = ""
        simulator_error = False
        if stage1_error:
            stage1_simulator_reply = "The assistant failed to produce clarifying questions."
            simulator_error = True
        else:
            stage1_simulator_prompt = self._build_simulator_prompt(sample, stage1_questions or "No questions were asked.")
            stage1_simulator_reply, sim_error = await self._call_judge_freeform(stage1_simulator_prompt)
            simulator_error = sim_error or (not stage1_simulator_reply.strip())
            if simulator_error and not stage1_simulator_reply:
                stage1_simulator_reply = "No user response was generated."

        qa_block = stage1_simulator_reply.strip() if stage1_simulator_reply else "No additional information was provided."
        stage2_prompt = self._build_stage2_prompt(sample, qa_block)
        stage2_response, stage2_thinking, stage2_truncated, stage2_error = await self._call_test_model(
            stage2_prompt, self.stage2_temperature
        )

        final_judge = await self._judge_final_answer(sample, stage2_response)
        is_correct = bool(final_judge.get("is_correct"))

        return {
            "id": sample_id,
            "stage1_prompt": stage1_prompt,
            "stage1_response": stage1_response,
            "stage1_thinking": stage1_thinking,
            "stage1_truncated": stage1_truncated,
            "stage1_error": stage1_error,
            "stage1_simulator_prompt": stage1_simulator_prompt if not stage1_error else None,
            "stage1_simulator_reply": stage1_simulator_reply,
            "simulator_error": simulator_error,
            "stage2_prompt": stage2_prompt,
            "stage2_response": stage2_response,
            "stage2_thinking": stage2_thinking,
            "stage2_truncated": stage2_truncated,
            "stage2_error": stage2_error,
            "final_judge": final_judge,
            "is_correct": is_correct
        }

    async def _call_test_model(self, prompt: str, temperature: float) -> Tuple[str, str, str, bool]:
        async with self.model_semaphore:
            try:
                response, thinking, truncated = await self.model.infer_async(
                    message=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    timeout=self.model_timeout
                )
                return response, thinking, truncated, False
            except Exception as exc:
                return f"Error: {exc}", "none", "error", True

    async def _call_judge_freeform(self, prompt: str) -> Tuple[str, bool]:
        async with self.judge_semaphore:
            try:
                response, _, _ = await self.judge_model.infer_async(
                    message=[{"role": "user", "content": prompt}],
                    max_tokens=self.judge_max_tokens,
                    temperature=self.simulator_temperature,
                    timeout=self.judge_timeout
                )
                return response, False
            except Exception:
                return "", True

    async def _judge_final_answer(self, sample: Dict, assistant_answer: str) -> Dict[str, Any]:
        prompt = FATA_FINAL_JUDGE_PROMPT \
            .replace("<ori_question>", sample.get("ori_question", "")) \
            .replace("<degraded_question>", sample.get("degraded_question") or sample.get("ori_question", "")) \
            .replace("<expected_answer>", sample.get("expected_answer", "")) \
            .replace("<assistant_answer>", assistant_answer or "")

        raw_response = ""
        parsed = {}
        attempt_used = self.judge_json_retries
        for attempt in range(1, self.judge_json_retries + 1):
            async with self.judge_semaphore:
                try:
                    raw_response, _, _ = await self.judge_model.infer_async(
                        message=[{"role": "user", "content": prompt}],
                        max_tokens=self.judge_max_tokens,
                        temperature=self.final_judge_temperature,
                        timeout=self.judge_timeout
                    )
                except Exception as exc:
                    raw_response = f"Error: {exc}"
            parsed = parse_json_to_dict(raw_response)
            if parsed:
                attempt_used = attempt
                break

        if not parsed:
            parsed = {
                "is_correct": False,
                "_parse_error": "JudgeJSONParseFailed",
                "_judge_raw_response": raw_response
            }
        parsed.setdefault("_judge_raw_response", raw_response)
        parsed.setdefault("_parse_attempts", attempt_used)
        return parsed

    def _build_stage1_prompt(self, sample: Dict) -> str:
        visible_question = sample.get("degraded_question") or sample.get("ori_question", "")
        return FATA_STAGE1_PROMPT_TEMPLATE.replace("<degraded_question>", visible_question)

    def _build_simulator_prompt(self, sample: Dict, assistant_questions: str) -> str:
        ori_question = sample.get("ori_question", "")
        degraded_question = sample.get("degraded_question") or ori_question
        degraded_info = sample.get("degraded_info") or "No additional context was provided."
        required_points_text = _format_required_points(sample.get("required_points") or [])
        prompt = FATA_STAGE1_SIMULATOR_PROMPT \
            .replace("<ori_question>", ori_question) \
            .replace("<degraded_question>", degraded_question) \
            .replace("<degraded_info>", degraded_info) \
            .replace("<required_points>", required_points_text) \
            .replace("<assistant_questions>", assistant_questions or "No questions were asked.")
        return prompt

    def _build_stage2_prompt(self, sample: Dict, qa_block: str) -> str:
        degraded_question = sample.get("degraded_question") or sample.get("ori_question", "")
        return FATA_STAGE2_PROMPT_TEMPLATE \
            .replace("<degraded_question>", degraded_question) \
            .replace("<qa_block>", qa_block)
