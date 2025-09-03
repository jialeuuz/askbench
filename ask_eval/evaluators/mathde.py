import re
from typing import Dict, List, Tuple
from ask_eval.evaluators.base_evaluator import BaseEvaluator
from sympy import simplify, Eq
import random
from fractions import Fraction

class MathDeEvaluator(BaseEvaluator):
    def __init__(self, model, eval_config: Dict):
        super().__init__(model, eval_config)
        
    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """格式化数学问题"""
        prompt = ""
        
        # 添加few-shot示例
        if train_data and self.shot > 0:
            examples = random.sample(train_data, min(self.shot, len(train_data)))
            for i, example in enumerate(examples, 1):
                prompt += f"# Example {i}\n"
                prompt += f"{example['problem']}\n"
                prompt += f"Answer: {example['answer']}\n\n"
        
        prompt += data["degraded_question"]
        if include_answer:
            prompt += f"\nAnswer: {data['answer']}"
        else:
            prompt += '\nFormat your response as follows: "The correct answer is boxed{insert answer here}".'
        return prompt

    def validate_answer(self, prediction: str, reference: str) -> bool:
        """验证数学答案"""
        # 标准化表达式
        prediction = str(prediction).strip().lower()
        reference = str(reference).strip().lower()
        prediction = self._normalize_expression(prediction)
        reference = self._normalize_expression(reference)
        
        # 直接字符串匹配
        if prediction == reference:
            return True
            
        # 数值等价性验证
        try:
            pred_num = float(simplify(prediction))
            ref_num = float(simplify(reference))
            if abs(pred_num - ref_num) < 1e-6:
                return True
        except:
            pass
            
        # 符号等价性验证
        try:
            eq = Eq(simplify(prediction), simplify(reference))
            if eq.simplify():
                return True
        except:
            pass
            
        # 数字匹配
        return self._compare_numbers(prediction, reference)

    def _normalize_expression(self, expr: str) -> str:
        # 去除空格和无关符号
        # 移除各种LaTeX修饰符
        expr = re.sub(r'\\(left|right|big|,\s*|quad)', '', expr)
        # 统一分数命令
        expr = re.sub(r'\\dfrac', r'\\frac', expr)
        # 提取括号内容
        expr = re.match(r'\((.*)\)', expr).group(1) if re.match(r'\((.*)\)', expr) else expr
        # 清除LaTeX环境符号
        expr = expr.replace('$', '').strip()

        # 处理带前导零的数字
        expr = re.sub(r'\b0+(\d+)', r'\1', expr)
        
        try:
            # 分数标准化（如3/4 → \frac{3}{4}）
            if "/" in expr:
                parts = expr.split("/")
                if len(parts) == 2:
                    frac = Fraction(int(parts[0]), int(parts[1]))
                    return f"{frac.numerator}/{frac.denominator}"
            
            # 代数表达式化简（需安装SymPy）
            sympy_expr = simplify(expr)
            return str(sympy_expr)
        except:
            return expr  # 无法解析时保留原始答案

    def _compare_numbers(self, response: str, answer: str) -> bool:
        """比较模型的回答和参考答案"""
        response_numbers = self._extract_numbers(response)
        answer_numbers = self._extract_numbers(answer)
        if response_numbers and answer_numbers:
            # 如果都能提取出数字，比较数字集合
            return response_numbers == answer_numbers
        else:
            # 否则，检查是否相互包含
            return (answer in response) or (response in answer)

    def _extract_numbers(self, text: str) -> set:
        """从文本中提取所有数字并返回一个集合"""
        return set(re.findall(r'\d+', text))