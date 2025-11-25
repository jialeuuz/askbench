from ..base_evaluator import BaseEvaluator
from ask_eval.evaluators.judge_mixin import JudgeEvaluatorMixin
from typing import Dict, List
import re

INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
GPQA_JUDGE_PROMPT = """
You are judging whether the assistant answered a multiple-choice GPQA question correctly.
Consider both the option letter and its meaning.
- If the assistant never picks a specific option, mark it incorrect.
- Treat semantically equivalent descriptions of the correct option as correct.

Question:
{question}

Choices:
{choices}

Ground-truth answer:
{expected_answer}

Assistant response:
{model_response}

Extracted option (if any):
{extracted_answer}

Reply with:
Reasoning: <short explanation>
```json
{{"reason": "<why>", "result": "correct" | "incorrect"}}
```
""".strip()


class GpqaEvaluator(JudgeEvaluatorMixin, BaseEvaluator):
    """评估输出格式的正确性"""
    def __init__(self, model, eval_config: Dict, judge_model=None, judge_config: Dict = None):
        super().__init__(model=model, eval_config=eval_config, judge_model=judge_model, judge_config=judge_config)

    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """格式化单个样例"""
        # 构建选项字符串
        # prompt = f"What is the correct answer to this question: {data['question']}"
        # prompt += f"\n\nChoices:\n(A) {data['choice1']}\n(B) {data['choice2']}\n(C) {data['choice3']}\n(D) {data['choice4']}"
        # prompt += f"\n\nFormat your response as follows: \"The correct answer is (insert answer here)\""
        
        # return prompt
        prompt = f"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{data['question']}

A) {data['choice1']}
B) {data['choice2']}
C) {data['choice3']}
D) {data['choice4']}
""".strip()
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """从响应中提取答案的通用方法"""
        if not response or response == "Error":
            return "Error"
        try:
            response = response.replace("**", "")
            patterns = [
                r"(?i)Answer\s*:\s*([^\n]+)",
                r"answer\s*[:：]\s*([0-9a-zA-Z/\-\+\.]+)",  # 英文标注
                r'Answer: \((.)\)', 
                r'answer: \((.)\)'
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    raw_ans = match.group(1).strip()
                    return raw_ans
                    
            print('未正则匹配出答案')
            return 'Error'  # 未找到答案的情况
            
        except Exception as e:
            print(f"提取答案时出错: {str(e)}")
            return 'Error'

    def reference_answer_for_record(self, sample: Dict) -> Dict[str, str]:
        letter = INDEX_TO_LETTER.get(sample.get("correct_index", -1), "")
        choice_key = f"choice{sample.get('correct_index', 0) + 1}"
        return {
            "letter": letter,
            "text": sample.get(choice_key, "")
        }

    def build_judge_prompt(self, sample: Dict, prompt: str, model_response: str, extracted_answer: str) -> str:
        choices_block = "\n".join(
            [
                f"{letter}) {sample.get(f'choice{idx + 1}', '')}"
                for idx, letter in INDEX_TO_LETTER.items()
            ]
        )
        expected_letter = INDEX_TO_LETTER.get(sample.get("correct_index", -1), "")
        choice_key = f"choice{sample.get('correct_index', 0) + 1}"
        expected_text = sample.get(choice_key, "")
        expected = f"{expected_letter}) {expected_text}"
        question_text = sample.get("question") or prompt
        return GPQA_JUDGE_PROMPT.format(
            question=question_text.strip(),
            choices=choices_block.strip(),
            expected_answer=expected.strip(),
            model_response=str(model_response).strip(),
            extracted_answer=str(extracted_answer),
        )
