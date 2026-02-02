# ask_eval/evaluators/base_evaluator.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import json
import os
from tqdm.asyncio import tqdm
import re

class BaseEvaluator:
    """Base class for evaluators."""
    def __init__(self, model, eval_config: Dict):
        self.model = model
        self.eval_config = eval_config
        self.max_concurrent = eval_config.get("max_concurrent")
        self.max_tokens = eval_config.get("max_tokens")
        self.temperature = eval_config.get("temperature")
        self.shot = eval_config.get("shot", 0)  # default: 0-shot
        self.top_k = eval_config.get("top_k", -1)
        self.top_p = eval_config.get("top_p", -1)
        
    def extract_answer(self, response: str) -> str:
        """Extract an answer string from a model response (best-effort)."""
        if not response or response == "Error":
            return "Error"
        try:
            response = response.replace("**", "")
            patterns = [
                r"\\boxed{([^{}]+)}",       # LaTeX boxed answer (no nested braces)
                r"\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}",  # LaTeX boxed answer (one-level nesting)
                r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)\}",   # LaTeX boxed answer (multi-level nesting)
                r"boxed{([^{}]+)}",       # without backslash
                r"boxed{(.*)}",           # catch-all
                r"\*\*\(([ABCD])\)\*\*",  # pattern like **(A)**
                r"The answer is\s*([0-9a-zA-Z/\-\+\.]+)",    # English full-sentence pattern
                r'answer is \((.)\)', 
                r"答案\s*[:：是为]\s*([0-9a-zA-Z/\-\+\.]+)",    # Chinese label
                r"答案\s*[:：是为]\s*\(([0-9a-zA-Z/\-\+\.]+)\)",    # Chinese label
                r"(?i)Answer\s*:\s*([^\n]+)",
                r"answer\s*[:：]\s*([0-9a-zA-Z/\-\+\.]+)",  # English label
                r'Answer: \((.)\)', 
                r'answer: \((.)\)', 
                r'answer \((.)\)', 
                r"=\s*([0-9a-zA-Z/\-\+\.]+)\s*$",           # answer after '='
                r"[:：]\s*([0-9a-zA-Z/\-\+\.]+)\s*$"
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    raw_ans = match.group(1).strip()
                    return raw_ans
                    
            print("Failed to extract an answer via regex.")
            return "Error"  # answer not found
            
        except Exception as e:
            print(f"Error while extracting answer: {str(e)}")
            return "Error"

    @abstractmethod
    def format_example(self, data: Dict, include_answer: bool = False, train_data: List[Dict] = None) -> str:
        """Format a single example into a prompt.

        Args:
            data: current example data
            include_answer: whether to include the answer
            train_data: few-shot examples
        """
        pass
        
    @abstractmethod
    def validate_answer(self, prediction: str, reference: str) -> bool:
        """Validate whether prediction matches reference."""
        pass
    
    async def validate_answer_async(self, prediction: str, reference: str) -> bool:
        """Async version of validate_answer (optional)."""
        pass

    async def infer_batch(self, test_data: List[Dict], train_data: List[Dict] = None) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Run batched inference and return responses."""
        questions = []
        for data in test_data:
            prompt = self.format_example(data, include_answer=False, train_data=train_data)
            questions.append(prompt)
        try:
            responses, thinking_processes, truncated_flags = await self.model.infer_batch_async(questions, self.max_tokens, self.temperature, self.max_concurrent)
        except Exception as e:
            print(f"API call failed: {str(e)}")
            responses = ["Error"] * len(questions)
            thinking_processes = ["none"] * len(questions)
            truncated_flags = ["none"] * len(questions)
        
        return responses, thinking_processes, truncated_flags, questions

    def evaluate_responses(self, args, test_data: List[Dict], responses: List[str], thinking_processes: List[str], truncated_flags: List[str], prompts: List[str]) -> tuple:
        """Evaluate responses and write result files."""
        cors = []  # correctness flags
        response_records = []
        
        # Extract answers
        responses_extract = [self.extract_answer(response) for response in responses]

        # Truncation statistics
        truncation_stats = {
            "not_truncated": 0,
            "truncated": 0,
            "none": 0
        }
        
        # Process results
        for i, (data, response, response_extract, thinking, truncated, prompt) in enumerate(zip(test_data, responses, responses_extract, thinking_processes, truncated_flags, prompts)):
            # Validate answer
            truncation_stats[truncated] = truncation_stats.get(truncated, 0) + 1
            cor = 1 if self.validate_answer(response_extract, data["expected_answer"]) else 0
            cors.append(cor)
            
            # Record result
            record = {
                "question": prompt,
                "response": response,
                "extracted_answer": response_extract,
                "answer": data["expected_answer"],
                "correct": cor,
                "thinking_process": thinking,
                "truncated": truncated
            }
            response_records.append(record)

        # Save detailed results
        output_file = os.path.join(args.save_dir, "api_responses.json")
        os.makedirs(args.save_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response_records, f, indent=2, ensure_ascii=False)

        # Compute accuracy
        acc = sum(cors) / len(cors)
        
        # Build log with accuracy and truncation stats
        log = f"Average accuracy: {acc:.3f}\n"
        log += "Truncation statistics:\n"
        for status, count in truncation_stats.items():
            percentage = count / len(responses) * 100
            log += f"- {status}: {count} ({percentage:.1f}%)\n"
        
        return acc, cors, log

    
