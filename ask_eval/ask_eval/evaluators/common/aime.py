import re
from typing import Dict, List, Tuple
from ask_eval.evaluators.math import MathEvaluator
from sympy import simplify, Eq
import random

class AimeEvaluator(MathEvaluator):
    def __init__(self, model, eval_config: Dict):
        super().__init__(model, eval_config)
        
    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """格式化数学问题"""
        if "ori_question" in data.keys():
            return data["ori_question"]
        else:
            return data["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed" + r"{}."
    
    def validate_answer(self, prediction: str, reference: str) -> bool:
        """验证数学答案"""
        # 先把答案转为数字
        if isinstance(reference, str):
            reference = int(reference)
        
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