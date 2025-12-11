from typing import Dict
from ask_eval.evaluators.math import MathEvaluator
from sympy import simplify, Eq
from ask_eval.utils.qwen_math import strip_string, math_equal
from ask_eval.evaluators.judge_mixin import JudgeEvaluatorMixin

MATH_JUDGE_PROMPT = """
You are grading whether the assistant solved a math problem correctly.
Carefully compare the assistant's final answer with the official solution.
- Treat algebraic expressions symbolically; values equivalent to the ground truth are correct.
- If the assistant never provides a final answer, treat it as incorrect.

Problem statement:
{question}

Ground-truth final answer:
{expected_answer}

Assistant response:
{model_response}

Reply with:
Reasoning: <short explanation>
```json
{{"reason": "<why the judgement was made>", "result": "correct" | "incorrect"}}
```
""".strip()

class Math500Evaluator(JudgeEvaluatorMixin, MathEvaluator):
    def __init__(self, model, eval_config: Dict, judge_model=None, judge_config: Dict = None):
        super().__init__(model=model, eval_config=eval_config, judge_model=judge_model, judge_config=judge_config)

    def validate_answer(self, prediction: str, reference: str) -> bool:
        """验证数学答案"""
        try:
            reference = strip_string(reference)
            result = math_equal(prediction, reference)
        except:
            result = False
        return result
    

    def extract_answer(self, pred_str):
        """直接返回模型原文，Judge 负责判分。"""
        if not pred_str or pred_str == "Error":
            return "Error"
        return str(pred_str).strip()

    def build_judge_prompt(self, sample: Dict, prompt: str, model_response: str, extracted_answer: str) -> str:
        question_text = sample.get("ori_question") or prompt
        expected = sample.get("expected_answer", "")
        return MATH_JUDGE_PROMPT.format(
            question=question_text.strip(),
            expected_answer=str(expected).strip(),
            model_response=str(model_response).strip(),
            extracted_answer=str(extracted_answer),
        )
