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
    """创建评估器实例"""
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
    """运行评估"""
    # 加载配置
    model_config = get_model_config(config)
    generate_config = get_generate_config(config)
    path_config = get_path_config(config)
    evaluator_config = get_evaluator_config(config) if config.has_section("evaluatorconfig") else {}
    evalset_name = config.get("evalset", "evalsetname")
    
    # 获取评估器特定配置
    specific_config = get_specific_config(config, evalset_name)
    if specific_config:
        print(f"已加载 {evalset_name} 特定配置: {specific_config}")
        generate_config.update(specific_config)  # 合并到 generate_config
    print('generate_config:\n\n')
    print(generate_config)
    
    # 获取尝试次数 n，默认为1
    n_attempts = int(generate_config.get("n_attempts", 1))
    print(f"评估每个问题将尝试 {n_attempts} 次")
    
    loader_class = LOADER_MAP.get(evalset_name)
    if not loader_class:
        raise ValueError(f"No data loader found for evalset: {evalset_name}")
    
    # 创建输出目录
    save_dir = os.path.join(
        path_config["save_dir"],
        evalset_name,
        model_config["task_name"]
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建模型和评估器
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
        print("正在创建Judge模型...")
        judge_model = create_model(judge_config, judge_config)

    evaluator = create_evaluator(evalset_name, model, generate_config, judge_model=judge_model, judge_config=judge_config)
    metric_label = getattr(evaluator, "metric_label", "准确率")
    
    # 创建数据加载器并加载数据
    data_dir = os.path.join(path_config["data_dir"], evalset_name)
    
    loader = loader_class(data_dir)
        
    data = loader.load_data()
    # 处理不同数据集返回格式
    if isinstance(data, dict) and "test" in data:
        test_data = data["test"]
        train_data = data.get("train", [])  # 如果没有train则为空列表
    else:
        test_data = data
        train_data = None
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"开始评估 {evalset_name} - {model_config['task_name']}")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 存储所有尝试的结果
    all_responses = []
    all_thinking_processes = []
    all_truncated_flags = []
    all_prompts = []
    all_cors = []
    
    final_log = ""
    # 多次运行模型推理
    for attempt in range(n_attempts):
        print(f"执行第 {attempt+1}/{n_attempts} 次推理...")
        
        # 获取模型响应
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
    
        # 保存这次评估的所有结果
        all_responses.append(responses)
        all_thinking_processes.append(thinking_processes)
        all_truncated_flags.append(truncated_flags)
        all_cors.append(cors)
        if attempt == 0:  # 只需要保存一次prompts
            all_prompts = prompts
        
        print(f"第 {attempt+1} 次推理{metric_label}: {acc:.4f}")
        final_log += f"第 {attempt+1} 次推理{metric_label}: {acc:.4f}\n{log}"
    
    # 计算最终指标
    # 重组数据为每个问题的所有尝试结果
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

    # 计算平均准确率
    average_acc = total_correct / total_evaluated if total_evaluated else 0
    
    # 记录每个问题的所有结果
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
        
        # 添加汇总指标
        if n_attempts > 1:
            valid_attempts = [val for val in attempt_values if val is not None]
            record["pass@1"] = (
                sum(valid_attempts) / len(valid_attempts) if valid_attempts else None
            )
        
        final_records.append(record)
    
    # 保存汇总结果
    summary_file = os.path.join(save_dir, "summary_results.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)
    
    # 生成最终日志
    if n_attempts > 1:
        pass_label = getattr(
            evaluator,
            "multi_attempt_metric_label",
            f"Pass@1 (尝试次数为{n_attempts}，平均准确率)"
        )
        final_log += f"{pass_label}: {average_acc:.4f}\n"
    else:
        final_log += f"{metric_label}: {average_acc:.4f}\n"
    
    # 统计所有尝试的截断情况
    all_truncated = [flag for attempt_flags in all_truncated_flags for flag in attempt_flags]
    truncation_stats = {}
    for status in set(all_truncated):
        count = all_truncated.count(status)
        percentage = count / len(all_truncated) * 100
        truncation_stats[status] = (count, percentage)
    
    final_log += "总体截断统计:\n"
    for status, (count, percentage) in truncation_stats.items():
        final_log += f"- {status}: {count} ({percentage:.1f}%)\n"
    
    # 记录结果
    end_time = datetime.now()
    duration = end_time - start_time
    
    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"评估集: {evalset_name}\n")
        f.write(f"任务名称: {model_config['task_name']}\n")
        
        if n_attempts > 1:
            pass_label = getattr(
                evaluator,
                "multi_attempt_metric_label",
                f"Pass@1 (尝试次数为{n_attempts}，平均准确率)"
            )
            f.write(f"{pass_label}: {average_acc:.4f}\n")
        else:
            f.write(f"{metric_label}: {average_acc:.4f}\n")
            
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {duration}\n")
        f.write(f"\n详细日志:\n{final_log}")
    
    print(f"评估完成，结果已保存到: {result_file}")
    
    if n_attempts > 1:
        pass_label = getattr(
            evaluator,
            "multi_attempt_metric_label",
            f"Pass@1 (尝试次数为{n_attempts}，平均准确率)"
        )
        print(f"{pass_label}: {average_acc:.4f}")
    else:
        print(f"{metric_label}: {average_acc:.4f}")
        
    print(f"总耗时: {duration}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置")
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(args.config))

if __name__ == "__main__":
    main()
