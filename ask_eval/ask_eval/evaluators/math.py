import re
from typing import Dict, List, Tuple
from ask_eval.evaluators.base_evaluator import BaseEvaluator
from sympy import simplify, Eq
import random
from fractions import Fraction

class MathEvaluator(BaseEvaluator):
    def __init__(self, model, eval_config: Dict):
        super().__init__(model, eval_config)
        
    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """Format a math problem into a prompt string."""
        prompt = ""
        
        # Add few-shot examples
        if train_data and self.shot > 0:
            examples = random.sample(train_data, min(self.shot, len(train_data)))
            for i, example in enumerate(examples, 1):
                prompt += f"# Example {i}\n"
                prompt += f"{example['problem']}\n"
                prompt += f"Answer: {example['answer']}\n\n"
                
        # Add the current problem
        if "problem" in data:
            prompt += data["problem"]
        else:
            prompt += data["ori_question"]
        # if include_answer:
        #     prompt += f"\nAnswer: {data['answer']}"
        # else:
        #     prompt += '\nFormat your response as follows: "The correct answer is boxed{insert answer here}".'
        return prompt

    def validate_answer(self, prediction: str, reference: str) -> bool:
        """Validate a math answer."""
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

    def _normalize_expression(self, expr: str) -> str:
        # Remove common LaTeX wrappers and irrelevant tokens
        expr = re.sub(r'\\(left|right|big|,\s*|quad)', '', expr)
        # Normalize fraction commands
        expr = re.sub(r'\\dfrac', r'\\frac', expr)
        # Strip outer parentheses
        expr = re.match(r'\((.*)\)', expr).group(1) if re.match(r'\((.*)\)', expr) else expr
        # Remove LaTeX math markers
        expr = expr.replace('$', '').strip()

        # Remove leading zeros in numbers
        expr = re.sub(r'\b0+(\d+)', r'\1', expr)
        
        try:
            # Normalize simple fractions (e.g., 3/4)
            if "/" in expr:
                parts = expr.split("/")
                if len(parts) == 2:
                    frac = Fraction(int(parts[0]), int(parts[1]))
                    return f"{frac.numerator}/{frac.denominator}"
            
            # Simplify algebraic expressions (requires SymPy)
            sympy_expr = simplify(expr)
            return str(sympy_expr)
        except:
            return expr  # Keep the original text if parsing fails

    def _compare_numbers(self, response: str, answer: str) -> bool:
        """Compare the model response and reference answer via extracted numbers."""
        response_numbers = self._extract_numbers(response)
        answer_numbers = self._extract_numbers(answer)
        if response_numbers and answer_numbers:
            # If both contain numbers, compare sets
            return response_numbers == answer_numbers
        else:
            # Otherwise, check substring containment
            return (answer in response) or (response in answer)

    def _extract_numbers(self, text: str) -> set:
        """Extract all digit sequences from text as a set."""
        return set(re.findall(r'\d+', text))
