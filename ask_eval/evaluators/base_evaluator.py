# ask_eval/evaluators/base_evaluator.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import json
import os
from tqdm.asyncio import tqdm
import re

class BaseEvaluator:
    """评估器基类"""
    def __init__(self, model, eval_config: Dict):
        self.model = model
        self.eval_config = eval_config
        self.max_concurrent = eval_config.get("max_concurrent")
        self.max_tokens = eval_config.get("max_tokens")
        self.temperature = eval_config.get("temperature")
        self.shot = eval_config.get("shot", 0)  # 默认为0-shot
        self.top_k = eval_config.get("top_k", -1)
        self.top_p = eval_config.get("top_p", -1)
        
    def extract_answer(self, response: str) -> str:
        """从响应中提取答案的通用方法"""
        if not response or response == "Error":
            return "Error"
        try:
            response = response.replace("**", "")
            patterns = [
                r"\\boxed{([^{}]+)}",       # LaTeX标准答案(不包含嵌套括号)
                r"\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}",  # LaTeX标准答案(支持一层嵌套)
                r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)\}",   # LaTeX标准答案(支持多层嵌套)
                r"boxed{([^{}]+)}",       # 不带反斜杠
                r"boxed{(.*)}",           # 通配
                r"\*\*\(([ABCD])\)\*\*",  # 对于格式 **(A)**
                r"The answer is\s*([0-9a-zA-Z/\-\+\.]+)",    # 英文完整句式
                r'answer is \((.)\)', 
                r"答案\s*[:：是为]\s*([0-9a-zA-Z/\-\+\.]+)",    # 中文标注
                r"答案\s*[:：是为]\s*\(([0-9a-zA-Z/\-\+\.]+)\)",    # 中文标注
                r"(?i)Answer\s*:\s*([^\n]+)",
                r"answer\s*[:：]\s*([0-9a-zA-Z/\-\+\.]+)",  # 英文标注
                r'Answer: \((.)\)', 
                r'answer: \((.)\)', 
                r'answer \((.)\)', 
                r"=\s*([0-9a-zA-Z/\-\+\.]+)\s*$",           # 等号后的答案
                r"[:：]\s*([0-9a-zA-Z/\-\+\.]+)\s*$"
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

    @abstractmethod
    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """格式化单个样例
        Args:
            data: 当前样例数据
            include_answer: 是否包含答案
            train_data: few-shot示例数据
        """
        pass
        
    @abstractmethod
    def validate_answer(self, prediction: str, reference: str) -> bool:
        """验证答案是否正确"""
        pass
    
    async def validate_answer_async(self, prediction: str, reference: str) -> bool:
        """异步验证答案是否正确"""
        pass

    async def infer_batch(self, test_data: List[Dict], train_data: List[Dict] = None) -> Tuple[List[str], List[str], List[str], List[str]]:
        """批量推理获取响应"""
        questions = []
        for data in test_data:
            prompt = self.format_example(data, include_answer=False, train_data=train_data)
            questions.append(prompt)
        try:
            responses, thinking_processes, truncated_flags = await self.model.infer_batch_async(questions, self.max_tokens, self.temperature, self.max_concurrent)
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            responses = ["Error"] * len(questions)
            thinking_processes = ["none"] * len(questions)
            truncated_flags = ["none"] * len(questions)
        
        return responses, thinking_processes, truncated_flags, questions

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
            cor = 1 if self.validate_answer(response_extract, data["answer"]) else 0
            cors.append(cor)
            
            # 记录结果
            record = {
                "question": prompt,
                "response": response,
                "extracted_answer": response_extract,
                "answer": data["answer"],
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

    async def evaluate_responses_async(self, args, test_data: List[Dict], responses: List[str], thinking_processes: List[str], truncated_flags: List[str], prompts: List[str], evaluator_config: Dict) -> tuple:
        pass

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
            cor = 1 if self.validate_answer(extracted_answer, record["answer"]) else 0
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
    