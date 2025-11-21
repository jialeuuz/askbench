import os
import asyncio
import argparse
from datetime import datetime
import configparser

from ask_eval.utils.config import (
    get_model_config,
    get_generate_config,
    get_path_config,
    get_evaluator_config
)
from ask_eval.utils.model_factory import create_model
from ask_eval.data.data_map import LOADER_MAP
from ask_eval.evaluators.evaluator_map import EVALUATOR_MAP


async def run_fata_evaluation(config):
    """Run the two-stage FATA evaluation with judge-driven feedback."""
    model_config = get_model_config(config)
    generate_config = get_generate_config(config)
    path_config = get_path_config(config)

    if not config.has_section("evaluatorconfig"):
        raise ValueError("FATA evaluation requires an [evaluatorconfig] section for the judge model.")
    evaluator_config = get_evaluator_config(config)

    n_attempts = int(generate_config.get("n_attempts", 1))
    evalset_name = config.get("evalset", "evalsetname")

    save_dir = os.path.join(
        path_config["save_dir"],
        evalset_name,
        model_config["task_name"]
    )
    os.makedirs(save_dir, exist_ok=True)

    tested_model = create_model(model_config, generate_config)
    judge_model = create_model(evaluator_config, evaluator_config)

    loader_class = LOADER_MAP.get(evalset_name)
    if not loader_class:
        raise ValueError(f"No data loader registered for {evalset_name}.")
    data_dir = os.path.join(path_config["data_dir"], evalset_name)
    loader = loader_class(data_dir)
    examples = loader.load_data()

    evaluator_class = EVALUATOR_MAP.get(evalset_name)
    if not evaluator_class:
        raise ValueError(f"No evaluator registered for {evalset_name}.")
    evaluator = evaluator_class(
        model=tested_model,
        eval_config=generate_config,
        judge_model=judge_model,
        judge_config=evaluator_config
    )

    start_time = datetime.now()
    attempt_accuracies = []
    attempt_logs = []

    for attempt_idx in range(n_attempts):
        print(f"\n---- Running FATA attempt {attempt_idx + 1}/{n_attempts} ----")
        attempt_save_dir = save_dir if n_attempts == 1 else os.path.join(save_dir, f"attempt_{attempt_idx + 1}")
        os.makedirs(attempt_save_dir, exist_ok=True)

        accuracy, _, log, _ = await evaluator.evaluate_dataset(
            args=argparse.Namespace(save_dir=attempt_save_dir),
            test_data=examples
        )

        attempt_accuracies.append(accuracy)
        attempt_logs.append(log)
        print(f"Attempt {attempt_idx + 1} accuracy: {accuracy:.4f}")

    end_time = datetime.now()
    duration = end_time - start_time
    average_accuracy = sum(attempt_accuracies) / len(attempt_accuracies) if attempt_accuracies else 0.0
    attempt_accuracy_str = ", ".join(f"{acc:.4f}" for acc in attempt_accuracies)

    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"评估集: {evalset_name}\n")
        f.write(f"任务名称: {model_config['task_name']}\n")
        if n_attempts > 1:
            f.write(f"平均准确率 (尝试次数为{n_attempts}): {average_accuracy:.4f}\n")
            f.write(f"各次尝试准确率: {attempt_accuracy_str}\n")
        else:
            f.write(f"最终准确率: {average_accuracy:.4f}\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {duration}\n\n")
        for idx, log in enumerate(attempt_logs):
            if n_attempts > 1:
                f.write(f"===== 第 {idx + 1} 次尝试日志 =====\n")
            f.write(log)
            if not log.endswith("\n"):
                f.write("\n")
            if n_attempts > 1 and idx != len(attempt_logs) - 1:
                f.write("\n")

    print(f"\nFATA evaluation complete. Results saved to: {result_file}")
    if n_attempts > 1:
        print(f"平均准确率: {average_accuracy:.4f}")
        print(f"各次尝试准确率: {attempt_accuracy_str}")
    else:
        print(f"最终准确率: {average_accuracy:.4f}")
    print(f"总耗时: {duration}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Run the FATA two-stage evaluation.")
    parser.add_argument("--config", required=True, help="Path to the configuration file (.ini).")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    config.read(args.config, encoding="utf-8")

    asyncio.run(run_fata_evaluation(config))


if __name__ == "__main__":
    main()
