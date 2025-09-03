from ask_eval.evaluators.base_evaluator import BaseEvaluator
from typing import Dict, List, Tuple
import json
import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

choices = ["A", "B", "C", "D"]

class MedQAEvaluator(BaseEvaluator):
    """MedMCQA评估器"""
    def __init__(self, model, eval_config: Dict):
        super().__init__(model, eval_config)
        self.few_shot_prompt = None
        
    def format_example(self, data: Dict, include_answer: bool = False) -> str:
        """格式化单个样例"""
        # 构建问题
        question = data['question'].strip()
            
        # 构建选项
        options = []
        if 'opa' in data and 'opb' in data and 'opc' in data and 'opd' in data:
            options.append(f"A. {str(data[f'opa']).strip()}")
            options.append(f"B. {str(data[f'opb']).strip()}")
            options.append(f"C. {str(data[f'opc']).strip()}")
            options.append(f"D. {str(data[f'opd']).strip()}")
        elif 'opa' in data and 'opb' in data and 'opc' in data:
            options.append(f"A. {str(data[f'opa']).strip()}")
            options.append(f"B. {str(data[f'opb']).strip()}")
            options.append(f"C. {str(data[f'opc']).strip()}")
        else:
            for i in range(4):
                choice = str(data[f'choice{i+1}']).strip()
                options.append(f"{choices[i]}. {choice}")
        # 组合prompt
        ask = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n"
        prompt = ask + question + "\n" + "\n".join(options)
        # prompt += f"\n答案：{data['answer']}\n\n"
            
        return prompt
        
    def extract_answer(self, response: str) -> str:
        """从响应中提取答案"""
        if not response or response == "Error":
            return "Error"
        response = response.strip()
        
        # 1. 直接匹配选项
        if response in choices:
            return response
            
        # 2. 匹配 "The answer is X" 模式
        # patterns = [
        #     (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        #     (r'答案(是|为)选项 ?([ABCD])', 2),
        #     (r'故?选择?：? ?([ABCD])', 1),
        #     (r'([ABCD]) ?选?项(是|为)?正确', 1),
        #     (r'正确的?选项(是|为) ?([ABCD])', 2),
        #     (r'答案(应该)?(是|为)([ABCD])', 3),
        #     (r'选项 ?([ABCD]) ?(是|为)?正确', 1),
        #     (r'选择答案 ?([ABCD])', 1),
        #     (r'答案?：?([ABCD])', 1),
        #     (r'([ABCD])(选?项)?是?符合题意', 1),
        #     (r'答案选项：? ?([ABCD])', 1),  # chatglm
        #     (r'答案(选项)?为(.*?)([ABCD])', 3),  # chatgpt
        #     (r'\*\*Answer:?\*\*\s*([ABCD])', 1),  # markdown格式
        #     (r'Answer:?\s*([ABCD])', 1),  # 普通Answer格式
        #     (r'The answer is\s*([ABCD])', 1),  # 完整句子格式
        #     (r'[Tt]he correct answer is\s*([ABCD])', 1),  # 另一种完整句子格式
        #     (r'\\boxed{([ABCD])}', 1),  # LaTeX boxed格式
        #     (r'([ABCD])(.*?)当选', 1),  # 原递归模式
        #     (r'([ABCD])(.*?)正确', 1),  # 原递归模式
        #     (r'[^不]是：? ?([ABCD])', 1),  # 原弱匹配模式
        # ]
        patterns = [
            # ==== 英文精确匹配 ====
            (r'[Tt]he answer is\s*([ABCD])(?:[\.\)\],\s]|$)', 1),  # 例：The answer is C.
            (r'[Tt]he correct answer is\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'\*\*Answer:?\*\*\s*([ABCD])(?:[\.\)\],\s]|$)', 1),  # markdown
            (r'Answer:?\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
            (r'\\boxed\{([ABCD])\}', 1),  # LaTeX

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
            (r'答案(选项)?为(.*?)([ABCD])(?:[\.\)\],\s]|$)', 3),  # chatgpt-style
            (r'([ABCD])(.*?)当选', 1),
            (r'([ABCD])(.*?)正确', 1),

            # ==== 特殊弱匹配（最后才用）====
            (r'[^不]是：?\s*([ABCD])(?:[\.\)\],\s]|$)', 1),
        ]
        for pattern, idx in patterns:
            m = re.search(pattern, response, re.M)
            if m:
                answer = m.group(idx)
                return answer
        
        # 3. 提取第一个出现的选项字母
        letter_pattern = r'[ABCD]'
        match = re.search(letter_pattern, response)
        if match:
            return match.group(0)
            
        return "Error"
        
    def validate_answer(self, prediction: str, reference: int) -> bool:
        if not prediction or prediction.strip().upper() == "ERROR":
            return False

        pred_norm = prediction.strip().upper()
        answer_map = {"A": 1, "B": 2, "C": 3, "D": 4}

        if pred_norm in answer_map:
            return answer_map[pred_norm] == reference

        return pred_norm == str(reference)

    def evaluate_responses(self, args, test_data: List[Dict], responses: List[str], thinking_processes: List[str], truncated_flags: List[str], prompts: List[str]) -> tuple:
        """评估响应结果"""
        cors = []  # 记录正确性
        response_records = []
        
        # 提取答案
        responses_extract = [self.extract_answer(response) for response in responses]

        # 统计截断情况
        truncation_stats = {
            "not_truncated": 0,
            "truncated": 0,
            "none": 0
        }
        
        # 处理结果
        for i, (data, response, response_extract, thinking, truncated, prompt) in enumerate(zip(test_data, responses, responses_extract, thinking_processes, truncated_flags, prompts)):
            # 验证答案
            truncation_stats[truncated] = truncation_stats.get(truncated, 0) + 1
            cor = 1 if self.validate_answer(response_extract, data["cop"]) else 0
            cors.append(cor)
            
            # 记录结果
            record = {
                "question": prompt,
                "response": response,
                "extracted_answer": response_extract,
                "answer": data["cop"],
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
        acc = sum(cors) / len(cors)
        
        # 生成日志，包含准确率和截断统计
        log = f"Average accuracy: {acc:.3f}\n"
        log += "Truncation statistics:\n"
        for status, count in truncation_stats.items():
            percentage = count / len(responses) * 100
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