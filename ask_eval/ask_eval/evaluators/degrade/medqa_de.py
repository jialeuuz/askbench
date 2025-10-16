from ask_eval.evaluators.base_evaluator import BaseEvaluator
from typing import Dict, List, Tuple
import json
import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

choices = ["A", "B", "C", "D"]

class MedQADeEvaluator(BaseEvaluator):
    """MedQA评估器"""
    def __init__(self, model, eval_config: Dict):
        super().__init__(model, eval_config)
        self.few_shot_prompt = None
        
    def format_example(self, data: Dict, include_answer: bool = False) -> str:
        """格式化单个样例"""
        # 构建问题，使用degraded_question字段
        question = data['degraded_question'].strip()
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
        
    def validate_answer(self, prediction: str, reference: str) -> bool:
        """
        验证答案是否正确。
        现在接收的reference是字符串形式的正确选项（如 "A"）。
        """
        # 检查预测或参考答案是否在提取时出错
        if not prediction or prediction.strip().upper() == "ERROR":
            return False
        if not reference or reference.strip().upper() == "ERROR":
            # 如果参考答案无法提取，说明数据有问题，也算作错误
            return False

        # 直接比较标准化后的字符串
        return prediction.strip().upper() == reference.strip().upper()

    def evaluate_responses(self, args, test_data: List[Dict], responses: List[str], thinking_processes: List[str], truncated_flags: List[str], prompts: List[str]) -> tuple:
        """评估响应结果"""
        cors = []  # 记录正确性
        response_records = []
        
        # 提取模型预测的答案
        responses_extract = [self.extract_answer(response) for response in responses]

        # 统计截断情况
        truncation_stats = {
            "not_truncated": 0,
            "truncated": 0,
            "none": 0
        }
        
        # 处理结果
        for i, (data, response, response_extract, thinking, truncated, prompt) in enumerate(zip(test_data, responses, responses_extract, thinking_processes, truncated_flags, prompts)):
            truncation_stats[truncated] = truncation_stats.get(truncated, 0) + 1
            
            # 从 data['answer'] 中提取标准答案（如 "A"）
            # 复用 extract_answer 函数来处理 "The answer is A." 这样的格式
            reference_answer_raw = data.get("expected_answer", "")
            reference_answer_extract = self.extract_answer(reference_answer_raw)

            # 验证答案
            cor = 1 if self.validate_answer(response_extract, reference_answer_extract) else 0
            cors.append(cor)
            
            # 记录结果
            record = {
                "question": prompt,
                "response": response,
                "extracted_answer": response_extract,
                "ground_truth_raw": reference_answer_raw, # 记录原始答案
                "ground_truth_extracted": reference_answer_extract, # 记录提取后的答案
                "correct": cor,
                "thinking_process": thinking,
                "truncated": truncated
            }
            response_records.append(record)

        # 保存详细结果
        output_file = os.path.join(args.save_dir, "api_responses.json")
        os.makedirs(args.save_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response_records, f, indent=2, ensure_ascii=False)

        # 计算准确率
        acc = sum(cors) / len(cors) if cors else 0
        
        # 生成日志，包含准确率和截断统计
        log = f"Average accuracy: {acc:.3f}\n"
        log += "Truncation statistics:\n"
        for status, count in truncation_stats.items():
            percentage = count / len(responses) * 100 if responses else 0
            log += f"- {status}: {count} ({percentage:.1f}%)\n"
        
        return acc, cors, log
        
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
