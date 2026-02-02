import re
from typing import Dict, List

from ask_eval.evaluators.base_evaluator import BaseEvaluator
from ask_eval.evaluators.judge_mixin import JudgeEvaluatorMixin

BBH_JUDGE_PROMPT = """
You are grading a Big-Bench Hard (BBH) question. Compare the assistant's final answer with the ground-truth answer.
- Accept the correct option letter when choices are provided, or an equivalent textual answer.
 - If multiple answers appear, focus on the explicit final answer (for example, after "Answer:").
- If the assistant never gives a definite answer, mark it incorrect.

{category_block}Question:
{question}

Ground-truth answer:
{expected_answer}

Assistant response:
{model_response}

Reply with:
Reasoning: <short explanation>
```json
{{"reason": "<why>", "result": "correct" | "incorrect"}}
```
""".strip()


class BBHEvaluator(JudgeEvaluatorMixin, BaseEvaluator):
    """Single-turn evaluator for BBH with judge-model scoring."""

    def __init__(self, model, eval_config: Dict, judge_model=None, judge_config: Dict = None):
        super().__init__(model=model, eval_config=eval_config, judge_model=judge_model, judge_config=judge_config)

    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """Return the prompt for the tested model."""
        question = data.get("ori_question") or data.get("question") or ""
        category = data.get("category")
        category_prefix = f"[Task: {category}]\n" if category else ""
        suffix = (
            "\n\nProvide only the final answer. If options are given, reply with the option letter; "
            "otherwise reply with the exact answer."
        )
        return f"{category_prefix}{question.strip()}{suffix}"

    def extract_answer(self, response: str) -> str:
        """Return the raw model output (no regex-based extraction)."""
        if not response or response == "Error":
            return "Error"
        return response.strip()

    def build_judge_prompt(self, sample: Dict, prompt: str, model_response: str, extracted_answer: str) -> str:
        """Build the judge prompt with ground truth and model reply."""
        question_text = sample.get("ori_question") or prompt
        expected = sample.get("expected_answer", "")
        category = sample.get("category")
        category_block = f"Category: {category}\n" if category else ""
        return BBH_JUDGE_PROMPT.format(
            category_block=category_block,
            question=question_text.strip(),
            expected_answer=str(expected).strip(),
            model_response=str(model_response).strip(),
            extracted_answer=str(extracted_answer),
        )
