import os
import asyncio
import argparse
from datetime import datetime
import configparser

from ask_eval.utils.config import (
    get_model_config,
    get_generate_config,
    get_path_config,
    get_evaluator_config,
)
from ask_eval.utils.model_factory import create_model
from ask_eval.data.data_map import LOADER_MAP
from ask_eval.evaluators.evaluator_map import EVALUATOR_MAP


async def run_ask_lone_evaluation(config):
    model_config = get_model_config(config)
    generate_config = get_generate_config(config)
    path_config = get_path_config(config)

    if not config.has_section('evaluatorconfig'):
        raise ValueError("AskLone 评测需要 [evaluatorconfig] 配置裁判模型。")
    evaluator_config = get_evaluator_config(config)

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
        raise ValueError(f"未在 LOADER_MAP 中找到数据加载器: {evalset_name}")
    data_dir = os.path.join(path_config["data_dir"], evalset_name)
    loader = loader_class(data_dir)
    examples = loader.load_data()

    evaluator_class = EVALUATOR_MAP.get(evalset_name)
    if not evaluator_class:
        raise ValueError(f"未在 EVALUATOR_MAP 中找到评测器: {evalset_name}")

    evaluator = evaluator_class(
        model=tested_model,
        eval_config=generate_config,
        judge_model=judge_model,
        judge_config=evaluator_config
    )

    start_time = datetime.now()
    print(f"开始评估 {evalset_name} - {model_config['task_name']}")

    avg_score, per_sample_scores, log, _ = await evaluator.evaluate_dataset(
        args=argparse.Namespace(save_dir=save_dir),
        test_data=examples
    )

    end_time = datetime.now()
    duration = end_time - start_time

    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"评估集: {evalset_name}\n")
        f.write(f"任务名称: {model_config['task_name']}\n")
        f.write(f"最终正确率: {avg_score:.4f}\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {duration}\n\n")
        f.write(log)
        if not log.endswith("\n"):
            f.write("\n")

    print(f"评估完成，结果已保存到: {result_file}")
    print(f"最终正确率: {avg_score:.4f}")
    print(log)
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Run AskLone evaluation.")
    parser.add_argument("--config", required=True, help="配置文件路径 (.ini)")
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    cfg.read(args.config, encoding="utf-8")

    asyncio.run(run_ask_lone_evaluation(cfg))


if __name__ == "__main__":
    main()
