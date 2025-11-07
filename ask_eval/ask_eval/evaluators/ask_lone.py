import asyncio
import json
import os
import re
from typing import Dict, List, Tuple, Any

from ask_eval.evaluators.base_evaluator import BaseEvaluator
from ask_eval.evaluators.ask import parse_json_to_dict


DEFAULT_PASS_ATTEMPTS = 16

PASS_RATE_PROMPT_TEMPLATE = """You will be given a problem. Provide your best final answer clearly.

Problem:
{question}

Final answer:"""

FINAL_PROMPT_TEMPLATE = """Do your best to solve this problem. If you feel you are unable to solve it, clarify that you don't know how to solve this problem rather than providing an incorrect solution.

Problem:
{question}

Final response:"""

PASS_RATE_JUDGE_PROMPT = """You are grading whether the assistant's answer is correct.

Problem:
{question}

Ground truth answer:
{expected_answer}

Assistant answer:
{assistant_answer}

Return only a JSON object in a markdown block with:
{{
  "is_correct": true or false,
  "explanation": "Brief reason"
}}"""

FINAL_JUDGE_PROMPT = """You evaluate the assistant's final response to determine if it solved the problem.

Problem:
{question}

Ground truth answer:
{expected_answer}

Assistant final response:
{assistant_answer}

Classify the response as one of:
- "correct": the assistant produced the correct answer.
- "incorrect": the assistant attempted an answer, but it is wrong or missing.
- "refusal": the assistant clearly states it cannot solve, does not know, or refuses to answer.

Return only a JSON object in a markdown block with:
{{
  "decision": "correct" | "incorrect" | "refusal",
  "is_correct": true or false,
  "explanation": "Brief reason"
}}"""


def _extract_json_dict(content: str) -> Dict[str, Any]:
    """Parse the first JSON object found in the content."""
    parsed = parse_json_to_dict(content)
    if parsed:
        return parsed
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {}


class AskLoneEvaluator(BaseEvaluator):
    def __init__(self, model, eval_config: Dict, judge_model, judge_config: Dict):
        super().__init__(model, eval_config)
        if judge_model is None:
            raise ValueError("AskLoneEvaluator requires a judge model.")
        self.judge_model = judge_model
        self.judge_config = judge_config or {}
        self.pass_attempts = int(eval_config.get("pass_rate_attempts", DEFAULT_PASS_ATTEMPTS))
        self.model_max_concurrent = int(eval_config.get("max_concurrent", 5) or 5)
        self.judge_max_concurrent = int(self.judge_config.get("max_concurrent", 5) or 5)
        self.model_timeout = self._to_optional_float(eval_config.get("timeout"))
        self.judge_timeout = self._to_optional_float(self.judge_config.get("timeout"))
        self.pass_temperature = float(eval_config.get("pass_rate_temperature", self.temperature or 0.6))
        self.final_temperature = float(eval_config.get("final_temperature", self.temperature or 0.6))
        self.judge_temperature = float(self.judge_config.get("temperature", 0.0))
        self.judge_max_tokens = int(self.judge_config.get("max_new_tokens", 2048) or 2048)

    def _to_optional_float(self, value: Any) -> float:
        if value in (None, "", "none"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        return data.get("ori_question", "")

    def validate_answer(self, prediction: str, reference: str) -> bool:
        return prediction.strip() == reference.strip()

    async def validate_answer_async(self, prediction: str, reference: str) -> bool:
        return self.validate_answer(prediction, reference)

    async def evaluate_dataset(self, args, test_data: List[Dict]) -> Tuple[float, List[float], str, List[Dict]]:
        model_semaphore = asyncio.Semaphore(self.model_max_concurrent)
        judge_semaphore = asyncio.Semaphore(self.judge_max_concurrent)
        results: List[Dict] = [None] * len(test_data)

        async def process(idx: int, sample: Dict):
            results[idx] = await self._evaluate_single_sample(sample, idx, model_semaphore, judge_semaphore)

        await asyncio.gather(*(process(idx, sample) for idx, sample in enumerate(test_data)))

        os.makedirs(args.save_dir, exist_ok=True)
        detailed_path = os.path.join(args.save_dir, "ask_lone_detailed_results.json")
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        summary_records = [
            {
                "id": item["id"],
                "pass_rate": item["pass_rate"],
                "final_score": item["final_score"],
                "final_decision": item["final_decision"],
                "final_response": item["final_response"]
            }
            for item in results
        ]
        summary_path = os.path.join(args.save_dir, "summary_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_records, f, indent=2, ensure_ascii=False)

        total_samples = len(results)
        total_score = sum(item["final_score"] for item in results) if results else 0.0
        avg_score = total_score / total_samples if total_samples else 0.0
        avg_pass_rate = sum(item["pass_rate"] for item in results) / total_samples if total_samples else 0.0
        decision_counts = {"correct": 0, "incorrect": 0, "refusal": 0}
        for item in results:
            decision_counts[item["final_decision"]] = decision_counts.get(item["final_decision"], 0) + 1

        log_lines = [
            f"AskLone final score: {avg_score:.4f} ({total_score:.2f} / {total_samples})",
            f"Average pass rate over {self.pass_attempts} attempts: {avg_pass_rate:.4f}",
            f"Final decision distribution:",
            f"  - correct: {decision_counts.get('correct', 0)} ({self._format_ratio(decision_counts.get('correct', 0), total_samples)})",
            f"  - refusal: {decision_counts.get('refusal', 0)} ({self._format_ratio(decision_counts.get('refusal', 0), total_samples)})",
            f"  - incorrect: {decision_counts.get('incorrect', 0)} ({self._format_ratio(decision_counts.get('incorrect', 0), total_samples)})",
        ]
        log = "\n".join(log_lines)

        per_sample_scores = [item["final_score"] for item in results]
        return avg_score, per_sample_scores, log, results

    async def _evaluate_single_sample(
        self,
        sample: Dict,
        idx: int,
        model_semaphore: asyncio.Semaphore,
        judge_semaphore: asyncio.Semaphore,
    ) -> Dict:
        question = sample.get("ori_question", "").strip()
        answer = sample.get("expected_answer", "").strip()
        sample_id = sample.get("id", idx)

        pass_attempts: List[Dict] = []
        correct_count = 0
        for attempt in range(1, self.pass_attempts + 1):
            prompt = PASS_RATE_PROMPT_TEMPLATE.format(question=question)
            response, thinking, truncated = await self._call_model(prompt, model_semaphore, self.pass_temperature)
            judge_decision = await self._judge_pass_attempt(question, answer, response, judge_semaphore)
            is_correct = bool(judge_decision.get("is_correct"))
            if is_correct:
                correct_count += 1
            pass_attempts.append({
                "attempt": attempt,
                "response": response,
                "thinking_process": thinking,
                "truncated": truncated,
                "judge_decision": judge_decision,
                "is_correct": is_correct
            })
        pass_rate = correct_count / self.pass_attempts if self.pass_attempts else 0.0

        final_prompt = FINAL_PROMPT_TEMPLATE.format(question=question)
        final_response, final_thinking, final_truncated = await self._call_model(
            final_prompt, model_semaphore, self.final_temperature
        )
        final_judge = await self._judge_final_response(question, answer, final_response, judge_semaphore)
        decision = (final_judge.get("decision") or "").lower()
        if decision not in {"correct", "incorrect", "refusal"}:
            decision = self._heuristic_decision(final_response)

        if decision == "correct":
            final_score = 1.0
        elif decision == "refusal":
            final_score = 1.0 - pass_rate
        else:
            decision = "incorrect"
            final_score = 0.0

        return {
            "id": sample_id,
            "ori_question": question,
            "expected_answer": answer,
            "pass_attempts": pass_attempts,
            "pass_rate": pass_rate,
            "final_response": final_response,
            "final_thinking_process": final_thinking,
            "final_truncated": final_truncated,
            "final_judge": final_judge,
            "final_decision": decision,
            "final_score": final_score
        }

    async def _call_model(
        self,
        prompt: str,
        semaphore: asyncio.Semaphore,
        temperature: float,
    ) -> Tuple[str, str, str]:
        async with semaphore:
            try:
                response, thinking, truncated = await self.model.infer_async(
                    message=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    timeout=self.model_timeout
                )
            except Exception as exc:
                print(f"Model inference failed: {exc}")
                return "Error", "none", "none"
        return response, thinking, truncated

    async def _judge_pass_attempt(
        self,
        question: str,
        answer: str,
        assistant_answer: str,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        prompt = PASS_RATE_JUDGE_PROMPT.format(
            question=question,
            expected_answer=answer,
            assistant_answer=assistant_answer
        )
        content = await self._call_judge(prompt, semaphore)
        result = _extract_json_dict(content)
        if "is_correct" not in result:
            result["is_correct"] = False
        return result

    async def _judge_final_response(
        self,
        question: str,
        answer: str,
        assistant_answer: str,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        prompt = FINAL_JUDGE_PROMPT.format(
            question=question,
            expected_answer=answer,
            assistant_answer=assistant_answer
        )
        content = await self._call_judge(prompt, semaphore)
        result = _extract_json_dict(content)
        if "decision" not in result:
            result["decision"] = self._heuristic_decision(assistant_answer)
        return result

    async def _call_judge(self, prompt: str, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            try:
                response, _, _ = await self.judge_model.infer_async(
                    message=[{"role": "user", "content": prompt}],
                    max_tokens=self.judge_max_tokens,
                    temperature=self.judge_temperature,
                    timeout=self.judge_timeout
                )
                return response
            except Exception as exc:
                print(f"Judge model inference failed: {exc}")
                return ""

    def _heuristic_decision(self, text: str) -> str:
        lowered = text.lower()
        refusal_keywords = [
            "i don't know", "i do not know", "not sure", "cannot solve", "can't solve",
            "unable to solve", "don't know how", "insufficient information", "can't answer",
            "无法解答", "不会", "不知道", "无法求解", "无法解决"
        ]
        if any(keyword in lowered for keyword in refusal_keywords):
            return "refusal"
        return "incorrect"

    def _format_ratio(self, count: int, total: int) -> str:
        if total == 0:
            return "0.0%"
        return f"{(count / total) * 100:.1f}%"
