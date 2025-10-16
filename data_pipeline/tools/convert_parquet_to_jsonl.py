import os
import pandas as pd

# 固定参数
input_dir = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/NuminaMath-CoT"  # 存放 parquet 文件的目录
output_file = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/NuminaMath_cot.jsonl"  # 输出的 jsonl 文件名

def convert_parquet_to_jsonl(input_dir, output_file):
    dfs = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".parquet"):
            file_path = os.path.join(input_dir, filename)
            print(f"正在读取: {file_path}")
            df = pd.read_parquet(file_path)
            dfs.append(df)
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.to_json(output_file, orient="records", lines=True, force_ascii=False)
        print(f"已保存到: {output_file}")
    else:
        print("未找到 parquet 文件。")

if __name__ == "__main__":
    convert_parquet_to_jsonl(input_dir, output_file)