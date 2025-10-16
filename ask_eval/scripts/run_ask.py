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
    
    # 从配置中获取最大对话轮次，提供一个默认值
    max_turns = config.getint("evaluatorconfig", "max_turns", fallback=5) # 默认5轮

    evalset_name = config.get("evalset", "evalsetname")
    
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

    # 4. 加载数据
    print(f"正在使用 LOADER_MAP 中的 '{evalset_name}' 加载器...")
    loader_class = LOADER_MAP.get(evalset_name)
    if not loader_class:
        raise ValueError(f"在 LOADER_MAP 中未找到 '{evalset_name}' 的加载器。")
    
    data_dir = os.path.join(path_config["data_dir"], evalset_name)
    loader = loader_class(data_dir)
    examples = loader.load_data()
    
    print(f"已加载 {len(examples)} 个评测样本。")
    print("数据格式要求: 'degraded_question', 'ori_question', 'expected_answer', 'degraded_info'")

    # 5. 实例化评测器
    print(f"正在使用 EVALUATOR_MAP 中的 '{evalset_name}' 评测器...")
    evaluator_class = EVALUATOR_MAP.get(evalset_name)
    if not evaluator_class:
        raise ValueError(f"在 EVALUATOR_MAP 中未找到 '{evalset_name}' 的评测器。")
        
    evaluator = evaluator_class(
        model=tested_model,
        eval_config=generate_config,
        judge_model=judge_model
    )

    # 记录开始时间
    start_time = datetime.now()
    print(f"\n开始评估 {evalset_name} - {model_config['task_name']}")

    # 6. [核心] 调用多轮对话评测方法
    final_accuracy, all_scores, log = await evaluator.evaluate_multi_turn(
        args=argparse.Namespace(save_dir=save_dir),
        test_data=examples,
        max_turns=max_turns
    )

    # 7. 记录最终结果
    end_time = datetime.now()
    duration = end_time - start_time

    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"评估集: {evalset_name}\n")
        f.write(f"任务名称: {model_config['task_name']}\n")
        f.write(f"最终正确率 (Accuracy): {final_accuracy:.4f}\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {duration}\n\n")
        f.write(log)

    print(f"\n评估完成，简洁结果已保存到: {result_file}")
    print(f"总耗时: {duration}")
    print("-" * 30)
    print(log)
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