import random
import os
from tqdm import tqdm

def sample_random_jsonl(input_file_path: str, output_file_path: str, sample_size: int, random_seed: int):
    """
    从指定的JSONL文件中随机采样指定数量的行，并保存到新文件。

    该函数采用两遍扫描法，内存效率高，适合处理大文件。

    Args:
        input_file_path (str): 输入的JSONL文件路径。
        output_file_path (str): 输出的采样后JSONL文件路径。
        sample_size (int): 需要采样的行数。
        random_seed (int): 随机种子，用于保证结果可复现。
    """
    print("--- 开始随机采样任务 ---")
    print(f"  [配置] 输入文件: {input_file_path}")
    print(f"  [配置] 输出文件: {output_file_path}")
    print(f"  [配置] 采样数量: {sample_size}")
    print(f"  [配置] 随机种子: {random_seed}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # --- 第一遍：计算总行数 ---
        print("\n[步骤 1/3] 正在计算输入文件总行数...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in tqdm(f, desc="计数中"))
        
        print(f"文件总行数: {total_lines}")

        if sample_size > total_lines:
            print(f"警告: 采样数量 ({sample_size}) 大于文件总行数 ({total_lines})。将采样所有行。")
            sample_size = total_lines

        # --- 第二遍：生成要采样的行号 ---
        print("\n[步骤 2/3] 正在生成随机行号...")
        random.seed(random_seed)
        # 使用 random.sample 高效地获取不重复的随机索引
        indices_to_sample = random.sample(range(total_lines), k=sample_size)
        # 将列表转换为集合，以获得O(1)的平均查找时间复杂度，极大提高效率
        indices_to_sample_set = set(indices_to_sample)
        print(f"已生成 {len(indices_to_sample_set)} 个唯一的待采样行号。")

        # --- 第三遍：读取、采样并写入文件 ---
        print("\n[步骤 3/3] 正在读取文件并写入采样数据...")
        lines_written = 0
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            # 使用tqdm显示采样进度
            for i, line in tqdm(enumerate(infile), total=total_lines, desc="采样中"):
                if i in indices_to_sample_set:
                    outfile.write(line)
                    lines_written += 1
        
        print(f"\n成功写入 {lines_written} 行到 '{output_file_path}'。")
        print("--- 采样任务完成 ---")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file_path}' 未找到。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

if __name__ == "__main__":
    INPUT_FILE = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/NuminaMath_cot_extra_success.jsonl'
    OUTPUT_FILE = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/test.jsonl'
    SAMPLE_SIZE = 2000
    RANDOM_SEED = 42

    # 调用采样函数
    sample_random_jsonl(
        input_file_path=INPUT_FILE,
        output_file_path=OUTPUT_FILE,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )