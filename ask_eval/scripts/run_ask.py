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
#  主运行函数
# =============================================================================

async def run_ask_evaluation(config):
    """运行 AskBench Judge驱动的多轮对话评估"""
    # 1. 加载配置
    model_config = get_model_config(config)
    generate_config = get_generate_config(config)
    path_config = get_path_config(config)
    
    if not config.has_section('evaluatorconfig'):
        raise ValueError("AskBench 评测需要 [evaluatorconfig] 部分来配置Judge模型和最大轮次。")
    evaluator_config = get_evaluator_config(config)

    # AskBench 同样支持多次尝试，默认尝试 1 次
    n_attempts = int(generate_config.get("n_attempts", 1))
    print(f"评估每个问题将尝试 {n_attempts} 次")
    
    # 从配置中获取最大对话轮次，提供一个默认值
    max_turns = config.getint("evaluatorconfig", "max_turns", fallback=3) # 默认3轮

    evalset_name = config.get("evalset", "evalsetname")
    generate_config["task_label"] = evalset_name
    
    # 2. 创建输出目录
    save_dir = os.path.join(
        path_config["save_dir"],
        evalset_name,
        model_config["task_name"]
    )
    os.makedirs(save_dir, exist_ok=True)

    # 3. 创建模型实例
    print("正在创建被测模型...")
    tested_model = create_model(model_config, generate_config)
    
    print("正在创建Judge模型 (兼任仲裁、评估、仿人角色)...")
    judge_model = create_model(evaluator_config, evaluator_config)

    # 3.1 创建 Simulator 模型（可选，默认复用 evaluatorconfig）
    if config.has_section("simulatorconfig"):
        simulator_config = get_simulator_config(config)
        print("正在创建Simulator模型 (用于仿人用户回合生成)...")
        simulator_model = create_model(simulator_config, simulator_config)
    else:
        simulator_config = evaluator_config
        print("未配置 [simulatorconfig]，将复用 Judge 模型作为 Simulator...")
        simulator_model = judge_model

    # 4. 加载数据
    print(f"正在使用 LOADER_MAP 中的 '{evalset_name}' 加载器...")
    loader_class = LOADER_MAP.get(evalset_name)
    if not loader_class:
        raise ValueError(f"在 LOADER_MAP 中未找到 '{evalset_name}' 的加载器。")
    
    data_dir = os.path.join(path_config["data_dir"], evalset_name)
    loader = loader_class(data_dir)
    examples = loader.load_data()
    
    print(f"已加载 {len(examples)} 个评测样本。")
    print("数据格式要求: 至少包含 'ori_question'、'expected_answer'，以及任务自带的场景字段（例如 'degraded_question'+'degraded_info'+'required_points'；或 'overconfidence_question'+'overconfidence_info'+('misleading_points'/'required_points')）。")

    # 5. 实例化评测器
    print(f"正在使用 EVALUATOR_MAP 中的 '{evalset_name}' 评测器...")
    evaluator_class = EVALUATOR_MAP.get(evalset_name)
    if not evaluator_class:
        raise ValueError(f"在 EVALUATOR_MAP 中未找到 '{evalset_name}' 的评测器。")
        
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

    # 记录开始时间
    start_time = datetime.now()
    print(f"\n开始评估 {evalset_name} - {model_config['task_name']}")

    attempt_accuracies = []
    attempt_logs = []

    for attempt_idx in range(n_attempts):
        print(f"\n---- 正在执行第 {attempt_idx + 1}/{n_attempts} 次尝试 ----")
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
        # all_scores 当前未用于汇总，但保留该返回以便后续扩展（如计算 pass@k）

        if final_accuracy is not None:
            attempt_accuracies.append(final_accuracy)
            print(f"第 {attempt_idx + 1} 次尝试准确率: {final_accuracy:.4f}")
        else:
            print(f"第 {attempt_idx + 1} 次尝试 Accuracy: 不适用（该任务不返回准确率）")

        if n_attempts > 1:
            attempt_result_file = os.path.join(attempt_save_dir, "results.txt")
            with open(attempt_result_file, 'w', encoding='utf-8') as f:
                f.write(f"评估集: {evalset_name}\n")
                f.write(f"任务名称: {model_config['task_name']}\n")
                f.write(f"尝试编号: {attempt_idx + 1}/{n_attempts}\n")
                if final_accuracy is not None:
                    f.write(f"AskBench Final Accuracy: {final_accuracy:.4f}\n")
                else:
                    f.write("AskBench Final Accuracy: N/A (该基准不提供 Accuracy)\n")
                f.write(f"开始时间: {attempt_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"结束时间: {attempt_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总耗时: {attempt_duration}\n\n")
                f.write(log)

    # 7. 记录最终结果
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
        f.write(f"评估集: {evalset_name}\n")
        f.write(f"任务名称: {model_config['task_name']}\n")
        if accuracy_available:
            if n_attempts > 1:
                f.write(f"平均准确率 (尝试次数为{n_attempts}): {average_accuracy:.4f}\n")
                f.write(f"各次尝试准确率: {attempt_accuracy_str}\n")
            else:
                f.write(f"最终正确率 (Accuracy): {average_accuracy:.4f}\n")
        else:
            f.write("Accuracy 指标: 本任务不返回该指标。\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {duration}\n\n")
        for idx, log in enumerate(attempt_logs):
            if n_attempts > 1:
                f.write(f"===== 第 {idx + 1} 次尝试日志 =====\n")
            f.write(log)
            if not log.endswith("\n"):
                f.write("\n")
            if idx != len(attempt_logs) - 1:
                f.write("\n")

    print(f"\n评估完成，汇总结果已保存到: {result_file}")
    if accuracy_available:
        if n_attempts > 1:
            print(f"平均准确率 (尝试次数为{n_attempts}): {average_accuracy:.4f}")
            print(f"各次尝试准确率: {attempt_accuracy_str}")
        else:
            print(f"最终正确率 (Accuracy): {average_accuracy:.4f}")
    else:
        print("Accuracy 指标: 本任务不返回该指标。")
    print(f"总耗时: {duration}")
    print("-" * 30)
    for idx, log in enumerate(attempt_logs):
        if n_attempts > 1:
            print(f"===== 第 {idx + 1} 次尝试日志 =====")
        print(log)
        if n_attempts > 1 and idx != len(attempt_logs) - 1:
            print("-" * 30)
    print("-" * 30)
    
    return save_dir

def main():
    """主入口函数"""
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
