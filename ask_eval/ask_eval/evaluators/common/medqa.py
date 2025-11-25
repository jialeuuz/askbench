from ask_eval.evaluators.base_evaluator import BaseEvaluator
from ask_eval.evaluators.judge_mixin import JudgeEvaluatorMixin
from typing import Dict, List, Tuple
import re

choices = ["A", "B", "C", "D"]
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

Extracted option (if any):
{extracted_answer}

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
        """从响应中提取答案"""
        if not response or response == "Error":
            return "Error"
        response = response.strip()
        
        # 1. 直接匹配选项
        if len(response) == 1 and response in choices:
            return response
            
        # 2. 匹配 "The answer is X" 等多种模式
        # 增加了对 "The answer is A." 这种格式的精确匹配
        patterns = [
            # ==== 英文精确匹配 ====
            (r'[Tt]he answer is\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'[Tt]he correct answer is\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'\*\*Answer:?\*\*\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'Answer:?\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'\\boxed\{([ABCD])\}', 1),

            # ==== 常见中英文直述结构 ====
            (r'答案(选项)?(是|为)：?\s*([ABCD])(?:[\.\)\],\s]|$)', 3),
            (r'答案(是|为)选项\s*([ABCD])(?:[\.\)\],\s]|$)', 2),
            (r'故?选择?：?\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'([ABCD])\s?选?项(是|为)?正确', 1),
            (r'正确的?选项(是|为)\s*([ABCD])(?:[\.\)\],\s]|$)', 2),
            (r'答案(应该)?(是|为)([ABCD])(?:[\.\)\],\s]|$)', 3),
            (r'选项\s*([ABCD])\s?(是|为)?正确', 1),
            (r'选择答案\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'答案?[:：]?\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'([ABCD])(选?项)?是?符合题意', 1),
            (r'答案选项：?\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'答案(选项)?为(.*?)([ABCD])(?:[\.\)\],\s]|$)', 3),
            (r'([ABCD])(.*?)当选', 1),
            (r'([ABCD])(.*?)正确', 1),

            # ==== 特殊弱匹配（最后才用）====
            (r'[^不]是：?\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
        ]
        for pattern, idx in patterns:
            m = re.search(pattern, response, re.M)
            if m:
                answer = m.group(idx).upper()
                return answer
        
        # 3. 提取第一个出现的选项字母
        letter_pattern = r'\b([ABCD])\b' # 使用\b确保是独立的字母
        match = re.search(letter_pattern, response)
        if match:
            return match.group(0)
            
        return "Error"
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
