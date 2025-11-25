import re
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

Extracted final answer (if any):
{extracted_answer}

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
        pred_str = pred_str.replace("\u043a\u0438", "")

        if "final answer is $" in pred_str and "$. I hope" in pred_str:
            # minerva_math
            tmp = pred_str.split("final answer is $", 1)[1]
            pred = tmp.split("$. I hope", 1)[0].strip()
        elif "boxed" in pred_str:
            ans = pred_str.split("boxed")[-1]
            if len(ans) == 0:
                a = ""
            elif ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split("$")[0].strip()
            pred = a
        elif "he answer is" in pred_str:
            pred = pred_str.split("he answer is")[-1].strip()
        elif "final answer is" in pred_str:
            pred = pred_str.split("final answer is")[-1].strip()
        elif "答案是" in pred_str:
            # Handle Chinese few-shot multiple choice problem answer extraction
            pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
        else:  
            # use the last number is False
            pred = ""

        # multiple line
        # pred = pred.split("\n")[0]
        pred = re.sub(r"\n\s*", "", pred)
        if pred != "" and pred[0] == ":":
            pred = pred[1:]
        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]
        if pred != "" and pred[-1] == "/":
            pred = pred[:-1]
        pred = strip_string(pred, skip_unit=False)
        return pred

    def build_judge_prompt(self, sample: Dict, prompt: str, model_response: str, extracted_answer: str) -> str:
        question_text = sample.get("ori_question") or prompt
        expected = sample.get("expected_answer", "")
        return MATH_JUDGE_PROMPT.format(
            question=question_text.strip(),
            expected_answer=str(expected).strip(),
            model_response=str(model_response).strip(),
            extracted_answer=str(extracted_answer),
        )
