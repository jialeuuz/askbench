import re
from typing import Dict, List, Tuple
from ask_eval.evaluators.math import MathEvaluator
from sympy import simplify, Eq
import random

class AimeEvaluator(MathEvaluator):
    def __init__(self, model, eval_config: Dict):
        super().__init__(model, eval_config)
        
    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """Format a math problem into a prompt string."""
        if "ori_question" in data.keys():
            return data["ori_question"]
        else:
            return data["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed" + r"{}."
    
    def validate_answer(self, prediction: str, reference: str) -> bool:
        """Validate a math answer."""
        # Convert reference to a number if possible
        if isinstance(reference, str):
            reference = int(reference)
        
        # Normalize expressions
        prediction = str(prediction).strip().lower()
        reference = str(reference).strip().lower()
        prediction = self._normalize_expression(prediction)
        reference = self._normalize_expression(reference)
        
        # Direct string match
        if prediction == reference:
            return True
            
        # Numeric equivalence check
        try:
            pred_num = float(simplify(prediction))
            ref_num = float(simplify(reference))
            if abs(pred_num - ref_num) < 1e-6:
                return True
        except:
            pass
            
        # Symbolic equivalence check
        try:
            eq = Eq(simplify(prediction), simplify(reference))
            if eq.simplify():
                return True
        except:
            pass
            
        # Fallback: compare extracted numbers
        return self._compare_numbers(prediction, reference)
