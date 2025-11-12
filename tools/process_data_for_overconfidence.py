import json
import random
from typing import Tuple, List, Optional
import pandas as pd
import os

def _process_single_record(data: dict, idx: int) -> List[dict]:
    """
    处理单条记录，返回处理后的样本列表
    """
    samples = []
    
    # 检查数据类型
    if 'conversation_history' in data:
        conversation = data['conversation_history']
        expected_answer = data.get('expected_answer', '')
        ori_question = data.get('ori_question', '')
        degraded_info = data.get('degraded_info', '')  # 可能存在的退化信息
        required_points = data.get('required_points', [])
        
        # 对于多轮对话，创建多个训练样本
        for i in range(0, len(conversation), 2):  # 每2个为一组(user, assistant)
            if i + 1 < len(conversation):  # 确保有assistant回复
                # 构建上下文：当前轮次之前的所有对话
                context = conversation[:i] if i > 0 else []
                
                # 构建到当前轮次的prompt
                prompt_messages = conversation[:i+1]  # 包含所有之前的对话和当前user输入
                answer = conversation[i+1]['content']  # assistant的回复
                
                # 判断是否是最后一轮
                is_final_turn = (i + 2 >= len(conversation))
                
                # verl格式 - 需要包含reward_model信息
                sample = {
                    "data_source": "overconfidence",
                    "prompt": prompt_messages,
                    "ability": "qa",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer  # 使用实际的assistant回复作为ground truth
                    },
                    "extra_info": {
                        "split": "train",  # 后续会更新
                        "index": idx * 100 + i,  # 创建唯一索引
                        "answer": answer,
                        "expected_answer": expected_answer if is_final_turn else "",
                        "ori_question": ori_question,
                        "degraded_info": degraded_info,
                        "required_points": required_points,
                        "is_final_turn": is_final_turn,
                        "question": prompt_messages[-1]['content'] if prompt_messages else "",
                        "context": context,  # 添加上下文信息
                        "turn_index": i // 2  # 添加轮次索引，方便追踪
                    }
                }
                samples.append(sample)
    else:
        # 处理简单QA数据（没有conversation_history的情况）
        question = data.get('question', data.get('ori_question', ''))
        answer = data.get('answer', data.get('expected_answer', ''))
        pass_rate = data.get('pass_rate', '')
        
        if question and answer:
            sample = {
                "data_source": "overconfidence",
                "prompt": [
                    {"role": "user", "content": question}
                ],
                "ability": "qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "answer": answer,
                    "expected_answer": answer,
                    "ori_question": question,
                    "degraded_info": "",
                    "is_final_turn": True,
                    "question": question,
                    "pass_rate": pass_rate,
                    "context": [],  # 单轮QA没有上下文
                    "turn_index": 0  # 单轮对话，轮次为0
                }
            }
            samples.append(sample)
    
    return samples


def _load_and_process_file(file_path: str) -> List[dict]:
    """
    加载并处理单个文件
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    processed_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                samples = _process_single_record(data, idx)
                processed_data.extend(samples)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {idx+1} 行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 第 {idx+1} 行处理失败: {e}")
                continue
    
    return processed_data


def _save_to_parquet(data: List[dict], output_dir: str, filename: str) -> str:
    """
    保存数据为parquet格式
    
    Args:
        data: 要保存的数据
        output_dir: 输出目录
        filename: 文件名
    
    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, filename)
    df.to_parquet(output_path, index=False)
    return output_path


def _print_data_statistics(train_data: List[dict], val_data: List[dict]):
    """
    打印数据统计信息
    """
    def get_stats(data: List[dict]) -> dict:
        multi_turn_count = sum(1 for item in data if item['extra_info'].get('turn_index', 0) > 0)
        single_turn_count = len(data) - multi_turn_count
        avg_context_len = sum(len(item['extra_info'].get('context', [])) for item in data) / len(data) if data else 0
        
        return {
            'total': len(data),
            'single_turn': single_turn_count,
            'multi_turn': multi_turn_count,
            'avg_context_len': avg_context_len
        }
    
    train_stats = get_stats(train_data)
    val_stats = get_stats(val_data)
    
    print(f"\n{'='*60}")
    print(f"数据统计信息")
    print(f"{'='*60}")
    print(f"\n训练集:")
    print(f"  总样本数: {train_stats['total']}")
    print(f"  单轮对话: {train_stats['single_turn']}")
    print(f"  多轮对话: {train_stats['multi_turn']}")
    print(f"  平均上下文长度: {train_stats['avg_context_len']:.2f}")
    
    print(f"\n验证集:")
    print(f"  总样本数: {val_stats['total']}")
    print(f"  单轮对话: {val_stats['single_turn']}")
    print(f"  多轮对话: {val_stats['multi_turn']}")
    print(f"  平均上下文长度: {val_stats['avg_context_len']:.2f}")
    
    print(f"\n总计:")
    print(f"  总样本数: {train_stats['total'] + val_stats['total']}")
    print(f"{'='*60}\n")


def process_data_for_grpo(
    train_input_file: str,
    train_parquet_dir: str,
    val_parquet_dir: str,
    val_input_file: Optional[str] = None,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[List, List]:
    """
    处理数据为GRPO格式，并划分训练集和验证集，保存为parquet格式
    
    Args:
        train_input_file: 训练数据输入文件路径
        train_parquet_dir: 训练集parquet输出目录
        val_parquet_dir: 验证集parquet输出目录
        val_input_file: 验证集输入文件路径（可选）。如果提供，则直接使用该文件作为验证集；
                       如果不提供，则从训练数据中按比例切割
        val_ratio: 验证集比例（仅在val_input_file为None时使用）
        random_seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        train_data, val_data: 处理后的训练集和验证集
    """
    random.seed(random_seed)
    
    # 处理训练数据
    if verbose:
        print(f"{'='*60}")
        print(f"开始处理数据")
        print(f"{'='*60}")
        print(f"正在处理训练数据: {train_input_file}")
    
    train_data = _load_and_process_file(train_input_file)
    
    if verbose:
        print(f"✓ 训练数据处理完成，共 {len(train_data)} 个样本")
    
    # 处理验证数据
    if val_input_file is not None:
        # 使用单独的验证集文件
        if verbose:
            print(f"\n正在处理验证数据: {val_input_file}")
        val_data = _load_and_process_file(val_input_file)
        if verbose:
            print(f"✓ 验证数据处理完成，共 {len(val_data)} 个样本")
    else:
        # 从训练数据中切割验证集
        if verbose:
            print(f"\n从训练数据中切割验证集（比例: {val_ratio}）")
        random.shuffle(train_data)
        val_size = int(len(train_data) * val_ratio)
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]
        if verbose:
            print(f"✓ 切割完成 - 训练集: {len(train_data)} 样本, 验证集: {len(val_data)} 样本")
    
    # 更新split标记
    for item in train_data:
        item['extra_info']['split'] = 'train'
    for item in val_data:
        item['extra_info']['split'] = 'test'
    
    # 保存为parquet
    if verbose:
        print(f"\n正在保存数据...")
    
    train_parquet_path = _save_to_parquet(train_data, train_parquet_dir, 'train.parquet')
    val_parquet_path = _save_to_parquet(val_data, val_parquet_dir, 'val.parquet')
    
    if verbose:
        print(f"✓ 训练集已保存至: {train_parquet_path}")
        print(f"✓ 验证集已保存至: {val_parquet_path}")
        
        # 打印详细统计信息
        _print_data_statistics(train_data, val_data)
    
    return train_data, val_data


def load_and_inspect_data(parquet_path: str, num_samples: int = 3):
    """
    加载并检查parquet数据，用于验证数据格式
    
    Args:
        parquet_path: parquet文件路径
        num_samples: 要展示的样本数量
    """
    df = pd.read_parquet(parquet_path)
    print(f"\n{'='*60}")
    print(f"数据检查: {parquet_path}")
    print(f"{'='*60}")
    print(f"总样本数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n前 {num_samples} 个样本:")
    
    for idx in range(min(num_samples, len(df))):
        print(f"\n--- 样本 {idx+1} ---")
        row = df.iloc[idx]
        print(f"Data Source: {row['data_source']}")
        print(f"Prompt: {row['prompt']}")
        print(f"Extra Info Keys: {list(row['extra_info'].keys())}")
        print(f"Context Length: {len(row['extra_info'].get('context', []))}")
        print(f"Context: {row['extra_info'].get('context', [])}")
        print(f"Question: {row['extra_info'].get('question', '')[:100]}...")
        print(f"Is Final Turn: {row['extra_info'].get('is_final_turn', False)}")
        print(f"Turn Index: {row['extra_info'].get('turn_index', 0)}")


# 使用示例
if __name__ == "__main__":
    # 示例1: 从训练数据切割验证集
    print("示例1: 从训练数据切割验证集")
    ori_data = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/data/overconfidence_34k/train_jsonl/overconfidence—med_and_math_34k_V2.jsonl'
    train_parquet_dir = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/data/overconfidence_34k/train_parquet'
    val_parquet_dir = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/data/overconfidence_34k/val_parquet'
    
    train_data, val_data = process_data_for_grpo(
        train_input_file=ori_data,
        train_parquet_dir=train_parquet_dir,
        val_parquet_dir=val_parquet_dir,
        val_ratio=0.1,
        random_seed=42,
        verbose=True
    )
    
    # 检查生成的数据
    print("\n检查训练集数据:")
    load_and_inspect_data(
        os.path.join(train_parquet_dir, 'train.parquet'),
        num_samples=2
    )
    
    print("\n检查验证集数据:")
    load_and_inspect_data(
        os.path.join(val_parquet_dir, 'val.parquet'),
        num_samples=2
    )
    
    # 示例2: 使用单独的验证集文件
    # print("\n" + "="*60)
    # print("示例2: 使用单独的验证集文件")
    # train_input = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/RL/data/17k_dapo_235b_reject/train_jsonl/17k_dapo_235b_reject.jsonl'
    # val_input = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/ask_eval/data/common/math500/test.jsonl'
    # train_parquet_dir = '/path/to/train_parquet'
    # val_parquet_dir = '/path/to/val_parquet'
    
    # train_data, val_data = process_data_for_grpo(
    #     train_input_file=train_input,
    #     train_parquet_dir=train_parquet_dir,
    #     val_parquet_dir=val_parquet_dir,
    #     val_input_file=val_input,  # 指定验证集文件
    #     random_seed=42,
    #     verbose=True
    # )