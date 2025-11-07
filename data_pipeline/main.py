import asyncio
import json
import os
import hashlib
from typing import List, Dict, Any, Generator, Set

# 导入我们自己的模块
from post_api import CustomAPI
from prompt_loader import load_prompts
import strategies # 假设你的策略函数都在 strategies.py 中

# ==============================================================================
# --- Pipeline 核心逻辑 (已优化) ---
# ==============================================================================

# ### 新增功能 1: 流式读取 JSONL 文件，避免内存爆炸 ###
def stream_jsonl(file_path: str, id_key: str = "id") -> Generator[Dict[str, Any], None, None]:
    """
    以流式方式（生成器）逐行读取和解析jsonl文件。
    如果数据行中缺少指定的 id_key，则为其生成一个确定性的哈希ID。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    # 如果缺少唯一ID，则根据内容生成一个，确保可重入性
                    if id_key not in item:
                        # 使用排序后的json字符串来确保哈希值是确定性的
                        deterministic_string = json.dumps(item, sort_keys=True)
                        item[id_key] = hashlib.sha256(deterministic_string.encode('utf-8')).hexdigest()
                    yield item
                except json.JSONDecodeError:
                    print(f"警告: 跳过无法解析的行: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"错误: 输入文件 '{file_path}' 未找到。程序终止。")
        exit()

# ### 新增功能 2: 获取已处理数据的ID ###
def get_processed_ids(file_path: str, id_key: str = "id") -> Set[Any]:
    """
    从输出文件中读取已成功处理的数据ID，用于断点重续。
    """
    processed_ids = set()
    if not os.path.exists(file_path):
        return processed_ids
    
    print(f"检测到已存在的输出文件 '{file_path}'，正在读取已处理的数据ID...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if id_key in item:
                    processed_ids.add(item[id_key])
            except json.JSONDecodeError:
                # 忽略输出文件中的损坏行
                continue
    print(f"找到 {len(processed_ids)} 个已处理的数据项。")
    return processed_ids


# ### 优化点 2 改动: save_jsonl 增加追加模式 ###
def save_jsonl(data: List[Dict[str, Any]], file_path: str, mode: str = 'w'):
    """
    将数据保存到jsonl文件。
    mode 'w': 覆盖写
    mode 'a': 追加写
    """
    if not data:
        # 在追加模式下，没有数据是正常情况，无需打印
        if mode == 'w':
            print(f"没有数据可保存到 '{file_path}'。")
        return
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    action = "保存" if mode == 'w' else "追加"
    print(f"成功将 {len(data)} 条数据{action}到 '{file_path}'。")


def _ensure_failure_meta(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """确保失败样本包含基本的失败元信息。"""
    ensured = []
    for it in items:
        if isinstance(it, dict) and '_failure' not in it:
            enriched = it.copy()
            enriched['_failure'] = {"step": "unknown", "reason": "unspecified", "attempts": 0}
            ensured.append(enriched)
        else:
            ensured.append(it)
    return ensured


async def main(
    strategy: str,
    input_file: str,
    output_file: str,
    prompts_file: str,
    api_urls: List[str],
    api_type: str,
    api_token: str,
    max_concurrent_requests: int,
    timeout: int,
    # ### 新增功能 3: 批处理大小和ID键名 ###
    batch_size: int,
    id_key: str = "id",
    # 是否重新处理历史失败项（默认 False 表示跳过失败项）
    reprocess_failed: bool = False,
):
    """
    数据构建 Pipeline 主函数，支持断点重续和分批处理。
    """
    print("--- Pipeline 开始执行 (支持断点重续) ---")
    print(f"策略: {strategy}")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print(f"批处理大小: {batch_size}")
    print("-----------------------------------------")

    # 1. 加载 Prompt 模板 (保持不变)
    templates = load_prompts(prompts_file)

    # 2. ### 优化点 3 改动: 实现断点重续逻辑 ###
    # 获取已处理的ID
    processed_ids = get_processed_ids(output_file, id_key)
    failed_base, failed_ext = os.path.splitext(output_file)
    failed_output_file = f"{failed_base}_failed{failed_ext}"
    # 失败的文件也需要检查，避免重复处理已失败的任务（可通过开关控制）
    failed_ids = get_processed_ids(failed_output_file, id_key)
    if reprocess_failed:
        print("已开启重新处理失败项：不会跳过失败ID。")
    else:
        processed_ids.update(failed_ids)
    
    if processed_ids:
        print(f"总计: 将跳过 {len(processed_ids)} 个已处理 (成功或失败) 的数据项。")
    # 统计成功与失败的历史数量，便于理解为何可能“瞬间结束”
    if os.path.exists(output_file):
        print(f"历史成功项数量: {len(get_processed_ids(output_file, id_key))}")
    if os.path.exists(failed_output_file):
        print(f"历史失败项数量: {len(failed_ids)}")

    # 3. 初始化 API 客户端 (保持不变)
    api_client = CustomAPI(
        url=api_urls[0],
        api_urls=api_urls,
        sk_token=api_token,
        api_type=api_type,
        timeout=timeout
    )

    # 4. 动态选择并获取策略函数 (保持不变)
    strategy_func = getattr(strategies, strategy, None)
    if not strategy_func or not callable(strategy_func):
        print(f"错误: 策略 '{strategy}' 在 strategies.py 中未找到或不是一个可调用函数。")
        available_strategies = [s for s in dir(strategies) if callable(getattr(strategies, s)) and not s.startswith("__")]
        print(f"可用策略: {available_strategies}")
        return

    # 5. ### 优化点 4 改动: 分批执行策略 ###
    print(f"正在执行策略 '{strategy}'，最大并发数: {max_concurrent_requests}...")
    
    original_infer_batch = api_client.infer_batch_async
    async def configured_infer_batch(*args, **kwargs):
        if 'max_concurrent' not in kwargs:
            kwargs['max_concurrent'] = max_concurrent_requests
        return await original_infer_batch(*args, **kwargs)
    api_client.infer_batch_async = configured_infer_batch
    
    batch_to_process = []
    total_processed_in_run = 0
    
    # 使用流式加载器
    for item in stream_jsonl(input_file, id_key):
        # 检查是否已处理
        if item.get(id_key) in processed_ids:
            continue
        
        batch_to_process.append(item)
        
        # 当批次达到指定大小时，执行处理
        if len(batch_to_process) >= batch_size:
            print(f"\n--- 正在处理一个 {len(batch_to_process)} 条数据的批次 ---")
            completed_data, failed_data = await strategy_func(
                api_client=api_client,
                data=batch_to_process,
                templates=templates
            )
            
            # 6. ### 优化点 5 改动: 追加保存结果 ###
            save_jsonl(completed_data, output_file, mode='a')
            if failed_data:
                save_jsonl(_ensure_failure_meta(failed_data), failed_output_file, mode='a')

            total_processed_in_run += len(batch_to_process)
            print(f"--- 批处理完成。本次运行已累计处理 {total_processed_in_run} 条新数据 ---")
            
            # 清空批次以备下一轮
            batch_to_process = []

    # 处理最后一批不足 batch_size 的数据
    if batch_to_process:
        print(f"\n--- 正在处理最后一批 {len(batch_to_process)} 条数据的批次 ---")
        completed_data, failed_data = await strategy_func(
            api_client=api_client,
            data=batch_to_process,
            templates=templates
        )

        save_jsonl(completed_data, output_file, mode='a')
        if failed_data:
            save_jsonl(_ensure_failure_meta(failed_data), failed_output_file, mode='a')

        total_processed_in_run += len(batch_to_process)
        print(f"--- 最后一批处理完成。 ---")

    print("\n--- Pipeline 执行完毕！ ---")
    print(f"本次运行共处理了 {total_processed_in_run} 条新数据。")
    final_success_count = len(get_processed_ids(output_file, id_key))
    final_failed_count = len(get_processed_ids(failed_output_file, id_key))
    print(f"输出文件 '{output_file}' 中现在总共有 {final_success_count} 条数据。")
    print(f"失败文件 '{failed_output_file}' 中现在总共有 {final_failed_count} 条数据。")


if __name__ == "__main__":
    # 可选策略（均返回: 成功样本列表, 失败样本列表）：
    # - "generate_degraded_question_and_info":
    #     生成劣化问题(degraded_question)、劣化信息(degraded_info)与缺失点清单(required_points)，
    #     常作为其他策略的前置步骤。
    # - "generate_multi_turn_training_data":
    #     生成完整多轮对话：追问 → 用户模拟 → 覆盖自检 → 最终答案 → Judge 分流 → 强制修正（必要时）。
    # - "strategy_direct_answer_and_correct":
    #     直接生成答案并判断；若错误则基于 expected_answer 与可选 solution 重构为“完美答案”，输出最终对话。
    STRATEGY = "generate_multi_turn_training_data"
    INPUT_FILE = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/sample_medmcqa_2k_clear.jsonl"
    OUTPUT_FILE = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/yitu/sample_medmcqa_2k_ask.jsonl"
    API_URLS = ["http://10.80.128.219:9012/v1/chat/completions"]
    API_TYPE = "default"
    API_TOKEN = "none"
    PROMPTS_FILE = "prompts.txt"
    MAX_CONCURRENT_REQUESTS = 200
    TIMEOUT = 3600
    BATCH_SIZE = 1000
    ID_KEY = "id" 
    # 可通过环境变量 REPROCESS_FAILED=1 打开（优先级高于下行常量）
    REPROCESS_FAILED = False  # 设置为 True 可重新处理历史失败项
    env_flag = os.getenv("REPROCESS_FAILED")
    if env_flag is not None:
        REPROCESS_FAILED = env_flag.strip() in ("1", "true", "True", "YES", "yes")
    
    asyncio.run(main(
        strategy=STRATEGY,
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        prompts_file=PROMPTS_FILE,
        api_urls=API_URLS,
        api_type=API_TYPE,
        api_token=API_TOKEN,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
        timeout=TIMEOUT,
        # 传递新增的配置
        batch_size=BATCH_SIZE,
        id_key=ID_KEY,
        reprocess_failed=REPROCESS_FAILED,
    ))
