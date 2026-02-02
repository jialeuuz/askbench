from ..base_evaluator import BaseEvaluator
from typing import Dict, List, Tuple
import json
import os
import numpy as np
import re

INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


class GpqaEvaluator(BaseEvaluator):
    """Evaluate whether the model output follows the expected answer format."""
    def __init__(self, model, eval_config: Dict):
        super().__init__(model, eval_config)

    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """Format a single example into a prompt."""
        # return prompt
        return data['degraded_question'].strip()
    
    def extract_answer(self, response: str) -> str:
        """Extract an answer string from a model response."""
        if not response or response == "Error":
            return "Error"
        try:
            response = response.replace("**", "")
            patterns = [
                r"(?i)Answer\s*:\s*([^\n]+)",
                r"answer\s*[:：]\s*([0-9a-zA-Z/\-\+\.]+)",  # English label
                r'Answer: \((.)\)', 
                r'answer: \((.)\)'
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    raw_ans = match.group(1).strip()
                    return raw_ans
                    
            print("Failed to extract an answer via regex.")
            return "Error"  # Answer not found
            
        except Exception as e:
            print(f"Error while extracting answer: {str(e)}")
            return "Error"

    def validate_answer(self, prediction: str, reference: Dict) -> bool:
        """Validate whether the extracted answer matches the expected format.

        Args:
            prediction: model prediction
            reference: reference answer / format requirement
        Returns:
            bool: whether the output matches the required format
        """
        prediction = prediction.strip().lower()
        reference = reference.strip().lower()
        extracted_answer = self.extract_answer(prediction)
        if not extracted_answer or extracted_answer.strip() == "":
            return False
        return reference in extracted_answer

    def evaluate_responses(self, args, test_data: List[Dict], responses: List[str], thinking_processes: List[str], truncated_flags: List[str], prompts: List[str]) -> tuple:
        """Evaluate responses and write result files."""
        cors = []  # correctness flags
        response_records = []
        
        # Truncation statistics
        truncation_stats = {
            "not_truncated": 0,
            "truncated": 0,
            "none": 0
        }
        
        answers_symbol = [INDEX_TO_LETTER[example['correct_index']] for example in test_data]
        answers = [example['choice{}'.format(example['correct_index']+1)] for example in test_data]
        
        # Evaluate each example
        for data, response, answer_symbol, answer, thinking, truncated, prompt in zip(test_data, responses, answers_symbol, answers, thinking_processes, truncated_flags, prompts):
            # Update truncation stats
            truncation_stats[truncated] = truncation_stats.get(truncated, 0) + 1
            
            # Validate answer format
            cor = 1 if self.validate_answer(response, answer_symbol) else 0
            cors.append(cor)
            
            # Record result
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

        # Save detailed results
        os.makedirs(args.save_dir, exist_ok=True)
        output_file = os.path.join(args.save_dir, "api_responses.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response_records, f, indent=2, ensure_ascii=False)

        # Compute accuracy
        acc = sum(cors) / len(cors)
        
        # Build log
        log = f"Format compliance rate: {acc:.3f}\n"
        log += "Truncation statistics:\n"
        for status, count in truncation_stats.items():
            percentage = count / len(responses) * 100
            log += f"- {status}: {count} ({percentage:.1f}%)\n"
        
        return acc, cors, log

    def reevaluate_responses(self, args):
        # Load response records from file
        input_file = os.path.join(args.save_dir, "api_responses.json")
        with open(input_file, 'r', encoding='utf-8') as f:
            response_records = json.load(f)

        # Updated correctness flags
        updated_cors = []

        # Update correct/extracted_answer fields
        updated_records = []
        for record in response_records:
            # Re-extract answer
            extracted_answer = self.extract_answer(record["response"])
            record["extracted_answer"] = extracted_answer
            
            # Re-validate answer
            cor = 1 if self.validate_answer(record["response"], record["answer_symbol"]) else 0
            record["correct"] = cor
            updated_cors.append(cor)
            updated_records.append(record)

        # Save updated results
        output_file = os.path.join(args.save_dir, "updated_api_responses.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_records, f, indent=2, ensure_ascii=False)
        
        # Compute updated accuracy
        new_acc = sum(updated_cors) / len(updated_cors)
        
        # Build log
        log = f"Updated format compliance rate: {new_acc:.3f}"
        
        # Return updated accuracy, correctness flags, and log
        return new_acc, updated_cors, log
