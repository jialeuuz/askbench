import os
import asyncio
import argparse
from datetime import datetime
import configparser
import inspect

from ask_eval.utils.config import (
    get_model_config, 
    get_generate_config, 
    get_path_config, 
    get_evaluator_config,
    get_simulator_config
)
from ask_eval.utils.model_factory import create_model
from ask_eval.data.data_map import LOADER_MAP
from ask_eval.evaluators.evaluator_map import EVALUATOR_MAP

# =============================================================================
# Main entry point
# =============================================================================

async def run_ask_evaluation(config):
    """Run AskBench Judge-driven multi-turn dialogue evaluation."""
    # 1) Load configs
    model_config = get_model_config(config)
    generate_config = get_generate_config(config)
    path_config = get_path_config(config)
    
    if not config.has_section('evaluatorconfig'):
        raise ValueError("AskBench evaluation requires an [evaluatorconfig] section (judge model + max_turns).")
    evaluator_config = get_evaluator_config(config)

    # AskBench also supports multiple attempts (default: 1).
    n_attempts = int(generate_config.get("n_attempts", 1))
    print(f"Each question will be evaluated with {n_attempts} attempt(s).")
    
    # Max dialogue turns (default: 3).
    max_turns = config.getint("evaluatorconfig", "max_turns", fallback=3)

    evalset_name = config.get("evalset", "evalsetname")
    generate_config["task_label"] = evalset_name
    
    # 2) Create output directory
    save_dir = os.path.join(
        path_config["save_dir"],
        evalset_name,
        model_config["task_name"]
    )
    os.makedirs(save_dir, exist_ok=True)

    # 3) Create model instances
    print("Creating tested model...")
    tested_model = create_model(model_config, generate_config)
    
    print("Creating judge model (arbiter / evaluator / user simulator)...")
    judge_model = create_model(evaluator_config, evaluator_config)

    # 3.1) Optional simulator model (defaults to evaluatorconfig)
    if config.has_section("simulatorconfig"):
        simulator_config = get_simulator_config(config)
        print("Creating simulator model (for synthetic user turns)...")
        simulator_model = create_model(simulator_config, simulator_config)
    else:
        simulator_config = evaluator_config
        print("No [simulatorconfig] provided; reusing judge model as simulator...")
        simulator_model = judge_model

    # 4) Load data
    print(f"Loading '{evalset_name}' via LOADER_MAP...")
    loader_class = LOADER_MAP.get(evalset_name)
    if not loader_class:
        raise ValueError(f"No loader found for '{evalset_name}' in LOADER_MAP.")
    
    data_dir = os.path.join(path_config["data_dir"], evalset_name)
    loader = loader_class(data_dir)
    examples = loader.load_data()
    
    print(f"Loaded {len(examples)} evaluation examples.")
    print(
        "Expected example fields: at minimum include 'ori_question' and 'expected_answer', plus task-specific "
        "scenario fields (e.g. 'degraded_question'+'degraded_info'+'required_points', or "
        "'overconfidence_question'+'overconfidence_info'+('misleading_points'/'required_points'))."
    )

    # 5) Instantiate evaluator
    print(f"Instantiating evaluator '{evalset_name}' from EVALUATOR_MAP...")
    evaluator_class = EVALUATOR_MAP.get(evalset_name)
    if not evaluator_class:
        raise ValueError(f"No evaluator found for '{evalset_name}' in EVALUATOR_MAP.")
        
    evaluator = evaluator_class(
        **{
            k: v
            for k, v in {
                "model": tested_model,
                "eval_config": generate_config,
                "judge_model": judge_model,
                "judge_config": evaluator_config,
                "simulator_model": simulator_model,
                "simulator_config": simulator_config,
            }.items()
            if k in inspect.signature(evaluator_class.__init__).parameters
        }
    )

    # Record start time
    start_time = datetime.now()
    print(f"\nStarting evaluation: {evalset_name} - {model_config['task_name']}")

    attempt_accuracies = []
    attempt_logs = []

    for attempt_idx in range(n_attempts):
        print(f"\n---- Running attempt {attempt_idx + 1}/{n_attempts} ----")
        attempt_start = datetime.now()
        attempt_save_dir = save_dir if n_attempts == 1 else os.path.join(save_dir, f"attempt_{attempt_idx + 1}")
        os.makedirs(attempt_save_dir, exist_ok=True)

        final_accuracy, all_scores, log = await evaluator.evaluate_multi_turn(
            args=argparse.Namespace(save_dir=attempt_save_dir),
            test_data=examples,
            max_turns=max_turns
        )

        attempt_end = datetime.now()
        attempt_duration = attempt_end - attempt_start

        attempt_logs.append(log)
        # all_scores is not aggregated yet, but kept for future extensions (e.g., pass@k).

        if final_accuracy is not None:
            attempt_accuracies.append(final_accuracy)
            print(f"Attempt {attempt_idx + 1} accuracy: {final_accuracy:.4f}")
        else:
            print(f"Attempt {attempt_idx + 1} accuracy: N/A (this benchmark does not return Accuracy)")

        if n_attempts > 1:
            attempt_result_file = os.path.join(attempt_save_dir, "results.txt")
            with open(attempt_result_file, 'w', encoding='utf-8') as f:
                f.write(f"Evalset: {evalset_name}\n")
                f.write(f"Task: {model_config['task_name']}\n")
                f.write(f"Attempt: {attempt_idx + 1}/{n_attempts}\n")
                if final_accuracy is not None:
                    f.write(f"AskBench Final Accuracy: {final_accuracy:.4f}\n")
                else:
                    f.write("AskBench Final Accuracy: N/A (this benchmark does not provide Accuracy)\n")
                f.write(f"Start time: {attempt_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End time: {attempt_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total time: {attempt_duration}\n\n")
                f.write(log)

    # 7) Write final results
    end_time = datetime.now()
    duration = end_time - start_time

    accuracy_available = len(attempt_accuracies) == n_attempts and n_attempts > 0
    average_accuracy = (
        sum(attempt_accuracies) / len(attempt_accuracies) if accuracy_available else None
    )
    attempt_accuracy_str = (
        ", ".join(f"{acc:.4f}" for acc in attempt_accuracies) if accuracy_available else ""
    )

    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Evalset: {evalset_name}\n")
        f.write(f"Task: {model_config['task_name']}\n")
        if accuracy_available:
            if n_attempts > 1:
                f.write(f"Average accuracy (n_attempts={n_attempts}): {average_accuracy:.4f}\n")
                f.write(f"Per-attempt accuracy: {attempt_accuracy_str}\n")
            else:
                f.write(f"Final accuracy: {average_accuracy:.4f}\n")
        else:
            f.write("Accuracy: N/A (this benchmark does not return Accuracy)\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {duration}\n\n")
        for idx, log in enumerate(attempt_logs):
            if n_attempts > 1:
                f.write(f"===== Attempt {idx + 1} log =====\n")
            f.write(log)
            if not log.endswith("\n"):
                f.write("\n")
            if idx != len(attempt_logs) - 1:
                f.write("\n")

    print(f"\nEvaluation complete. Summary saved to: {result_file}")
    if accuracy_available:
        if n_attempts > 1:
            print(f"Average accuracy (n_attempts={n_attempts}): {average_accuracy:.4f}")
            print(f"Per-attempt accuracy: {attempt_accuracy_str}")
        else:
            print(f"Final accuracy: {average_accuracy:.4f}")
    else:
        print("Accuracy: N/A (this benchmark does not return Accuracy)")
    print(f"Total time: {duration}")
    print("-" * 30)
    for idx, log in enumerate(attempt_logs):
        if n_attempts > 1:
            print(f"===== Attempt {idx + 1} log =====")
        print(log)
        if n_attempts > 1 and idx != len(attempt_logs) - 1:
            print("-" * 30)
    print("-" * 30)
    
    return save_dir

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Judge-driven multi-turn evaluation for AskBench.")
    parser.add_argument("--config", required=True, help="Path to the configuration file (.ini format)")
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    config.read(args.config, encoding='utf-8')
    
    asyncio.run(run_ask_evaluation(config))

if __name__ == "__main__":
    main()
