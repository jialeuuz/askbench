def merge_jsonl(input_paths, output_path):
    """
    Merge one or more JSONL files into a single output file.

    Args:
        input_paths (list): list of input file paths
        output_path (str): output file path
    """
    print(f"Merging {len(input_paths)} file(s) into {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for path in input_paths:
            try:
                with open(path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                print(f"- Added: {path}")
            except FileNotFoundError:
                print(f"- Warning: file not found, skipped: {path}")
    print("Done.")


if __name__ == '__main__':
    file1 = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/final_train_data/degrade_med_40k_oss120b_low.jsonl'
    file2 = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_math_40k_oss120b_low.jsonl'
    output_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/final_train_data/degrade_math_med_80k_oss120b_low.jsonl'
    files_to_merge = [file1, file2]
    output_filename = output_path
    merge_jsonl(files_to_merge, output_filename)
