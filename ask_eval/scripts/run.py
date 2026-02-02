# scripts/run.py
import os
import asyncio
import argparse
import inspect
from datetime import datetime
from typing import Dict
import json

from ask_eval.utils.config import load_config, get_model_config, get_generate_config, get_path_config, get_evaluator_config, get_specific_config
from ask_eval.utils.model_factory import create_model

from ask_eval.evaluators.evaluator_map import EVALUATOR_MAP
from ask_eval.data.data_map import LOADER_MAP


def create_evaluator(evalset_name: str, model, generate_config: Dict, judge_model=None, judge_config: Dict = None):
    """Create an evaluator instance."""
    evaluator_class = EVALUATOR_MAP.get(evalset_name)
    if not evaluator_class:
        raise ValueError(f"Unknown evalset: {evalset_name}")
    requires_judge = getattr(evaluator_class, "requires_judge", False)
    if requires_judge:
        if judge_model is None:
            raise ValueError(f"Evaluator {evalset_name} requires a judge model but none was provided.")
        return evaluator_class(model, generate_config, judge_model=judge_model, judge_config=judge_config)
    return evaluator_class(model, generate_config)

async def run_evaluation(config):
    """Run evaluation for a single task config."""
    # Load configs
    model_config = get_model_config(config)
    generate_config = get_generate_config(config)
    path_config = get_path_config(config)
    evaluator_config = get_evaluator_config(config) if config.has_section("evaluatorconfig") else {}
    evalset_name = config.get("evalset", "evalsetname")
    
    # Apply evalset-specific overrides (if present)
    specific_config = get_specific_config(config, evalset_name)
    if specific_config:
        print(f"Loaded evalset-specific overrides for {evalset_name}: {specific_config}")
        generate_config.update(specific_config)  # merge into generate_config
    print("generate_config:\n")
    print(generate_config)
    
    # Number of attempts per question (default: 1)
    n_attempts = int(generate_config.get("n_attempts", 1))
    print(f"Each question will be evaluated with {n_attempts} attempt(s).")
    
    loader_class = LOADER_MAP.get(evalset_name)
    if not loader_class:
        raise ValueError(f"No data loader found for evalset: {evalset_name}")
    
    # Create output directory
    save_dir = os.path.join(
        path_config["save_dir"],
        evalset_name,
        model_config["task_name"]
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model and evaluator
    model = create_model(model_config, generate_config)

    evaluator_class = EVALUATOR_MAP.get(evalset_name)
    if not evaluator_class:
        raise ValueError(f"Unknown evalset: {evalset_name}")
    requires_judge = getattr(evaluator_class, "requires_judge", False)

    judge_model = None
    judge_config = None
    if requires_judge:
        if not evaluator_config:
            raise ValueError(f"Evaluator {evalset_name} requires [evaluatorconfig] settings.")
        judge_config = evaluator_config
        print("Creating judge model...")
        judge_model = create_model(judge_config, judge_config)

    evaluator = create_evaluator(evalset_name, model, generate_config, judge_model=judge_model, judge_config=judge_config)
    metric_label = getattr(evaluator, "metric_label", "Accuracy")
    
    # Create data loader and load data
    data_dir = os.path.join(path_config["data_dir"], evalset_name)
    
    loader = loader_class(data_dir)
        
    data = loader.load_data()
    # Handle different dataset return formats
    if isinstance(data, dict) and "test" in data:
        test_data = data["test"]
        train_data = data.get("train", [])  # empty list if train split not provided
    else:
        test_data = data
        train_data = None
    
    # Record start time
    start_time = datetime.now()
    print(f"Starting evaluation: {evalset_name} - {model_config['task_name']}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Store results for all attempts
    all_responses = []
    all_thinking_processes = []
    all_truncated_flags = []
    all_prompts = []
    all_cors = []
    
    final_log = ""
    # Run multiple inference attempts
    for attempt in range(n_attempts):
        print(f"Running inference attempt {attempt+1}/{n_attempts}...")
        
        # Get model responses
        responses, thinking_processes, truncated_flags, prompts = await evaluator.infer_batch(test_data, train_data)
        
        evaluation_args = dict(
            args=argparse.Namespace(save_dir=os.path.join(save_dir, f"attempt_{attempt+1}")),
            test_data=test_data,
            responses=responses,
            thinking_processes=thinking_processes,
            truncated_flags=truncated_flags,
            prompts=prompts
        )
        evaluate_async = getattr(evaluator, "evaluate_responses_async", None)
        if evaluate_async and inspect.iscoroutinefunction(evaluate_async):
            acc, cors, log = await evaluate_async(**evaluation_args)
        else:
            acc, cors, log = evaluator.evaluate_responses(**evaluation_args)
    
        # Save this attempt's outputs
        all_responses.append(responses)
        all_thinking_processes.append(thinking_processes)
        all_truncated_flags.append(truncated_flags)
        all_cors.append(cors)
        if attempt == 0:  # prompts are identical across attempts
            all_prompts = prompts
        
        print(f"Attempt {attempt+1} {metric_label}: {acc:.4f}")
        final_log += f"Attempt {attempt+1} {metric_label}: {acc:.4f}\n{log}"
    
    # Compute final metrics
    # Re-group per-question results across attempts
    question_cors = []
    total_correct = 0
    total_evaluated = 0
    for q_idx in range(len(test_data)):
        q_cors = [all_cors[attempt][q_idx] for attempt in range(n_attempts)]
        question_cors.append(q_cors)
        for value in q_cors:
            if value is not None:
                total_correct += value
                total_evaluated += 1

    # Compute average accuracy across all evaluated items
    average_acc = total_correct / total_evaluated if total_evaluated else 0
    
    # Record per-question outputs for all attempts
    final_records = []
    for q_idx, data in enumerate(test_data):
        record = {
            "question": all_prompts[q_idx],
            "answer": data.get("answer", ""),
            "attempts": []
        }
        
        attempt_values = question_cors[q_idx]
        for attempt in range(n_attempts):
            cor_value = attempt_values[attempt]
            attempt_record = {
                "response": all_responses[attempt][q_idx],
                "thinking_process": all_thinking_processes[attempt][q_idx],
                "truncated": all_truncated_flags[attempt][q_idx],
                "correct": cor_value,
                "attempt_num": attempt + 1
            }
            record["attempts"].append(attempt_record)
        
        # Add aggregated metrics
        if n_attempts > 1:
            valid_attempts = [val for val in attempt_values if val is not None]
            record["pass@1"] = (
                sum(valid_attempts) / len(valid_attempts) if valid_attempts else None
            )
        
        final_records.append(record)
    
    # Save summary JSON
    summary_file = os.path.join(save_dir, "summary_results.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)
    
    # Build final log
    if n_attempts > 1:
        pass_label = getattr(
            evaluator,
            "multi_attempt_metric_label",
            f"Pass@1 (n_attempts={n_attempts}, average accuracy)"
        )
        final_log += f"{pass_label}: {average_acc:.4f}\n"
    else:
        final_log += f"{metric_label}: {average_acc:.4f}\n"
    
    # Truncation stats across all attempts
    all_truncated = [flag for attempt_flags in all_truncated_flags for flag in attempt_flags]
    truncation_stats = {}
    for status in set(all_truncated):
        count = all_truncated.count(status)
        percentage = count / len(all_truncated) * 100
        truncation_stats[status] = (count, percentage)
    
    final_log += "Overall truncation stats:\n"
    for status, (count, percentage) in truncation_stats.items():
        final_log += f"- {status}: {count} ({percentage:.1f}%)\n"
    
    # Write result file
    end_time = datetime.now()
    duration = end_time - start_time
    
    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Evalset: {evalset_name}\n")
        f.write(f"Task: {model_config['task_name']}\n")
        
        if n_attempts > 1:
            pass_label = getattr(
                evaluator,
                "multi_attempt_metric_label",
                f"Pass@1 (n_attempts={n_attempts}, average accuracy)"
            )
            f.write(f"{pass_label}: {average_acc:.4f}\n")
        else:
            f.write(f"{metric_label}: {average_acc:.4f}\n")
            
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {duration}\n")
        f.write(f"\nDetailed log:\n{final_log}")
    
    print(f"Evaluation complete. Results saved to: {result_file}")
    
    if n_attempts > 1:
        pass_label = getattr(
            evaluator,
            "multi_attempt_metric_label",
            f"Pass@1 (n_attempts={n_attempts}, average accuracy)"
        )
        print(f"{pass_label}: {average_acc:.4f}")
    else:
        print(f"{metric_label}: {average_acc:.4f}")
        
    print(f"Total time: {duration}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file (.ini)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    asyncio.run(run_evaluation(config))

if __name__ == "__main__":
    main()
