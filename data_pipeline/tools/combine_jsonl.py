def merge_jsonl(input_paths, output_path):
    """
    将一个或多个 JSONL 文件合并到一个输出文件中。

    Args:
        input_paths (list): 包含输入文件路径的列表。
        output_path (str): 输出文件的路径。
    """
    print(f"正在将 {len(input_paths)} 个文件合并到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for path in input_paths:
            try:
                with open(path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                print(f"- 已添加: {path}")
            except FileNotFoundError:
                print(f"- 警告: 文件未找到，已跳过: {path}")
    print("合并完成。")


if __name__ == '__main__':
    file1 = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/final_train_data/degrade_med_40k_oss120b_low.jsonl'
    file2 = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_math_40k_oss120b_low.jsonl'
    output_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/final_train_data/degrade_math_med_80k_oss120b_low.jsonl'
    files_to_merge = [file1, file2]
    output_filename = output_path
    merge_jsonl(files_to_merge, output_filename)