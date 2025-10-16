# main_queue.py

import sys
import json
import multiprocessing
import asyncio
from datetime import datetime

# 假设你的主逻辑文件名为 main.py
# 我们从中导入 main 函数，并重命名以避免混淆
try:
    from main import main as run_pipeline_task
except ImportError:
    print("错误: 无法从 'main.py' 导入 'main' 函数。")
    print("请确保 'main_queue.py' 和 'main.py' 在同一个目录下。")
    sys.exit(1)

# ==============================================================================
# --- 全局固定配置 (如果某些参数在所有任务中都一样，可以放在这里) ---
# ==============================================================================
# 这些参数将作为默认值，但可以被任务JSON中的同名键覆盖
FIXED_PARAMS = {
    "API_TYPE": "default",
    "API_TOKEN": "none",
    "PROMPTS_FILE": "prompts.txt",
    "MAX_CONCURRENT_REQUESTS": 200,
    "TIMEOUT": 600
}

def get_current_time():
    """获取格式化的当前时间字符串"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def worker_process(task_queue: list, worker_id: int):
    """
    每个并行工作进程执行的函数。
    它会按顺序串行执行分配给它的所有任务。
    
    :param task_queue: 一个任务字典的列表，例如 [{'STRATEGY': 's1', ...}, {'STRATEGY': 's2', ...}]
    :param worker_id: 工作进程的编号，用于日志输出。
    """
    print(f"[{get_current_time()}] [工作进程 {worker_id}] 已启动，包含 {len(task_queue)} 个串行任务。")
    
    for i, task_config in enumerate(task_queue):
        task_num = i + 1
        print("-" * 80)
        print(f"[{get_current_time()}] [工作进程 {worker_id}] 开始执行串行任务 {task_num}/{len(task_queue)}...")
        
        # 合并固定参数和任务特定参数，任务特定参数会覆盖固定参数
        params = FIXED_PARAMS.copy()
        params.update(task_config)

        # 检查必要参数是否存在
        required_keys = ['STRATEGY', 'INPUT_FILE', 'OUTPUT_FILE', 'API_URLS']
        if not all(key in params for key in required_keys):
            print(f"[{get_current_time()}] [工作进程 {worker_id}] 错误: 任务 {task_num} 配置不完整，缺少必要键。跳过此任务。")
            print(f"    必需键: {required_keys}")
            print(f"    当前配置: {task_config}")
            continue

        try:
            # run_pipeline_task 是一个 async 函数, 需要用 asyncio.run() 来执行
            asyncio.run(run_pipeline_task(
                strategy=params['STRATEGY'],
                input_file=params['INPUT_FILE'],
                output_file=params['OUTPUT_FILE'],
                prompts_file=params['PROMPTS_FILE'],
                api_urls=params['API_URLS'],
                api_type=params['API_TYPE'],
                api_token=params['API_TOKEN'],
                max_concurrent_requests=params['MAX_CONCURRENT_REQUESTS'],
                timeout=params['TIMEOUT']
            ))
            print(f"[{get_current_time()}] [工作进程 {worker_id}] 成功完成串行任务 {task_num}/{len(task_queue)}。")
        except Exception as e:
            print(f"[{get_current_time()}] [工作进程 {worker_id}] 错误: 串行任务 {task_num}/{len(task_queue)} 执行失败！")
            print(f"    错误详情: {e}")
            # 即使一个任务失败，也会继续执行队列中的下一个任务
    
    print(f"[{get_current_time()}] [工作进程 {worker_id}] 所有串行任务已处理完毕。")


if __name__ == "__main__":
    # 设置多进程启动方式为 'spawn'，在某些系统上更稳定
    multiprocessing.set_start_method("spawn", force=True)

    if len(sys.argv) != 2:
        print("用法: python main_queue.py '<JSON_CONFIG_STRING>'")
        print("示例: python main_queue.py '[[{\"STRATEGY\": ...}]]'")
        sys.exit(1)

    json_config_string = sys.argv[1]
    
    try:
        parallel_queues = json.loads(json_config_string)
        if not isinstance(parallel_queues, list):
            raise ValueError("JSON配置必须是一个列表。")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"错误: 解析JSON配置失败。请检查格式。")
        print(f"    错误详情: {e}")
        print(f"    接收到的字符串: {json_config_string}")
        sys.exit(1)

    num_parallel_workers = len(parallel_queues)
    if num_parallel_workers == 0:
        print("配置为空，没有任务需要执行。")
        sys.exit(0)

    print("=" * 80)
    print(f"任务调度器启动，将创建 {num_parallel_workers} 个并行工作进程。")
    print("=" * 80)

    # 为每个工作进程准备参数，包含任务队列和工作进程ID
    worker_args = [(queue, i + 1) for i, queue in enumerate(parallel_queues)]

    # 创建并启动进程池
    # with 语句确保进程池在完成后能被正确关闭和清理
    with multiprocessing.Pool(processes=num_parallel_workers) as pool:
        # starmap 会将 worker_args 中的每个元组解包作为 worker_process 函数的参数
        pool.starmap(worker_process, worker_args)

    print("=" * 80)
    print(f"[{get_current_time()}] 所有并行工作进程均已完成。任务调度结束。")
    print("=" * 80)