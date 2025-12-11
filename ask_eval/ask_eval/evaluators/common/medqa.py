from ask_eval.evaluators.base_evaluator import BaseEvaluator
from ask_eval.evaluators.judge_mixin import JudgeEvaluatorMixin
from typing import Dict, List, Tuple
MEDQA_JUDGE_PROMPT = """
You are judging whether a multiple-choice medical answer is correct.
The assistant must pick the same option as the ground-truth answer.
- If the assistant never states a final option, mark it incorrect.
- Ignore extra explanation and focus on the chosen option.

Question:
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

class MedQAEvaluator(JudgeEvaluatorMixin, BaseEvaluator):
    """MedQA评估器"""
    def __init__(self, model, eval_config: Dict, judge_model=None, judge_config: Dict = None):
        super().__init__(model=model, eval_config=eval_config, judge_model=judge_model, judge_config=judge_config)
        self.few_shot_prompt = None
        
    def format_example(self, data: Dict, include_answer: bool = False) -> str:
        """格式化单个样例"""
        # 构建问题，使用degraded_question字段
        question = data['ori_question'].strip()
        prompt = question
        # 如果需要构建few-shot prompt，可以取消下面的注释并传入include_answer=True
        # if include_answer:
        #     prompt += f"\n答案：{data['answer']}\n\n"
            
        return prompt
        
    def extract_answer(self, response: str) -> str:
        """直接返回模型原文，判分交给裁判模型。"""
        if not response or response == "Error":
            return "Error"
        return response.strip()
    async def infer_batch(self, test_data: List[Dict], train_data: List[Dict] = None) -> Tuple[List[str], List[str], List[str], List[str]]:
        """批量推理"""
        prompts = []
        
        # 处理测试样例
        for data in test_data:
            # 构建完整prompt
            prompt = self.format_example(data)
            prompts.append(prompt)
            
        # 获取响应
        try:
            responses, thinking_processes, truncated_flags = await self.model.infer_batch_async(prompts, self.max_tokens, self.temperature, self.max_concurrent)
        except Exception as e:
            print(f"Error generating response: {e}")
            responses = ["Error"] * len(prompts)
            thinking_processes = ["none"] * len(prompts)
            truncated_flags = ["none"] * len(prompts)
                
        return responses, thinking_processes, truncated_flags, prompts

    def build_judge_prompt(self, sample: Dict, prompt: str, model_response: str, extracted_answer: str) -> str:
        question_text = sample.get("ori_question") or prompt
        expected = sample.get("expected_answer", "")
        return MEDQA_JUDGE_PROMPT.format(
            question=question_text.strip(),
            expected_answer=str(expected).strip(),
            model_response=str(model_response).strip(),
            extracted_answer=str(extracted_answer),
        )
