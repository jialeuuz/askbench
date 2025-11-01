from ..base_evaluator import BaseEvaluator
from typing import Dict, List, Tuple
import json
import os
import numpy as np
import re

INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


class GpqaEvaluator(BaseEvaluator):
    """评估输出格式的正确性"""
    def __init__(self, model, eval_config: Dict):
        super().__init__(model, eval_config)

    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """格式化单个样例"""
        # return prompt
        return data['degraded_question'].strip()
    
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

    def validate_answer(self, prediction: str, reference: Dict) -> bool:
        """验证答案格式是否正确
        Args:
            prediction: 模型预测的答案
            reference: 参考答案格式要求
        Returns:
            bool: 是否符合格式要求
        """
        prediction = prediction.strip().lower()
        reference = reference.strip().lower()
        extracted_answer = self.extract_answer(prediction)
        if not extracted_answer or extracted_answer.strip() == "":
            return False
        return reference in extracted_answer

    def evaluate_responses(self, args, test_data: List[Dict], responses: List[str], thinking_processes: List[str], truncated_flags: List[str], prompts: List[str]) -> tuple:
        """评估响应结果"""
        cors = []  # 记录正确性
        response_records = []
        
        # 统计截断情况
        truncation_stats = {
            "not_truncated": 0,
            "truncated": 0,
            "none": 0
        }
        
        answers_symbol = [INDEX_TO_LETTER[example['correct_index']] for example in test_data]
        answers = [example['choice{}'.format(example['correct_index']+1)] for example in test_data]
        
        # 评估每个样本
        for data, response, answer_symbol, answer, thinking, truncated, prompt in zip(test_data, responses, answers_symbol, answers, thinking_processes, truncated_flags, prompts):
            # 统计截断情况
            truncation_stats[truncated] = truncation_stats.get(truncated, 0) + 1
            
            # 验证答案
            cor = 1 if self.validate_answer(response, answer_symbol) else 0
            cors.append(cor)
            
            # 记录结果
            record = {
                "question": prompt,
                "response": response,
                "answer_symbol": answer_symbol,
                "answer": answer,
                "correct": cor,
                "thinking_process": thinking,
                "truncated": truncated
            }
            response_records.append(record)

        # 保存详细结果
        os.makedirs(args.save_dir, exist_ok=True)
        output_file = os.path.join(args.save_dir, "api_responses.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response_records, f, indent=2, ensure_ascii=False)

        # 计算准确率
        acc = sum(cors) / len(cors)
        
        # 生成日志
        log = f"Format compliance rate: {acc:.3f}\n"
        log += "Truncation statistics:\n"
        for status, count in truncation_stats.items():
            percentage = count / len(responses) * 100
            log += f"- {status}: {count} ({percentage:.1f}%)\n"
        
        return acc, cors, log

    def reevaluate_responses(self, args):
        # 从文件中读取响应记录
        input_file = os.path.join(args.save_dir, "api_responses.json")
        with open(input_file, 'r', encoding='utf-8') as f:
            response_records = json.load(f)

        # 存储更新后的正确性结果
        updated_cors = []

        # 更新correct和extracted_answer字段
        updated_records = []
        for record in response_records:
            # 重新提取答案
            extracted_answer = self.extract_answer(record["response"])
            record["extracted_answer"] = extracted_answer
            
            # 重新验证答案
            cor = 1 if self.validate_answer(record["response"], record["answer_symbol"]) else 0
            record["correct"] = cor
            updated_cors.append(cor)
            updated_records.append(record)

        # 保存更新后的结果
        output_file = os.path.join(args.save_dir, "updated_api_responses.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_records, f, indent=2, ensure_ascii=False)
        
        # 计算新的准确率
        new_acc = sum(updated_cors) / len(updated_cors)
        
        # 生成日志
        log = f"Updated format compliance rate: {new_acc:.3f}"
        
        # 返回新的准确率、正确性列表和日志
        return new_acc, updated_cors, log