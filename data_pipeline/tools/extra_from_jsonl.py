import json
import os

def process_jsonl(input_file: str, output_file: str, columns_to_extract: list, column_rename_map: dict):
    """
    处理JSONL文件，根据指定列进行提取和重命名。

    :param input_file: 输入的JSONL文件路径。
    :param output_file: 输出的JSONL文件路径。
    :param columns_to_extract: 一个列表，包含需要从每行JSON中提取的键（列名）。
    :param column_rename_map: 一个字典，键是原始列名，值是新的列名。如果为空，则不进行重命名。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在。")
        return

    print(f"开始处理文件: {input_file}")
    print(f"将提取以下列: {columns_to_extract}")
    if column_rename_map:
        print(f"将进行如下重命名: {column_rename_map}")
    else:
        print("不进行列重命名。")

    # 使用 'with' 语句确保文件能被正确关闭
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # 逐行读取输入文件
        for line in infile:
            # 去除行尾的空白符（如换行符）
            line = line.strip()
            if not line:
                continue

            try:
                # 将JSON字符串解析为Python字典
                data = json.loads(line)
                
                # 创建一个新的字典来存储提取和重命名后的数据
                new_record = {}
                
                # 遍历需要提取的列名列表
                for original_key in columns_to_extract:
                    # 检查原始数据中是否存在这个键
                    if original_key in data:
                        # 使用 .get(key, default) 方法来确定新的键名
                        # 如果 original_key 在 rename_map 中，则使用其对应的值作为新键名
                        # 否则，使用 original_key 本身作为新键名
                        new_key = column_rename_map.get(original_key, original_key)
                        
                        # 将数据（原始值和新键）添加到新记录中
                        new_record[new_key] = data[original_key]
                
                # 如果新记录不为空，则将其转换为JSON字符串并写入输出文件
                if new_record:
                    # ensure_ascii=False 保证中文字符能被正确写入，而不是被转义
                    outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print(f"警告: 跳过格式错误的JSON行 -> {line}")

    print(f"处理完成！结果已保存至: {output_file}")


if __name__ == '__main__':
    
    # --- 1. 指定输入和输出文件路径 ---
    input_file_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/rubric/mindeval/data/common_language/math/aime2025/test.jsonl'
    output_file_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/ask_eval/data/common/aime2025/test_clear.jsonl'

    # --- 2. 指定要提取的列名 ---
    # 这是一个列表，包含所有你希望从原始文件中保留的列的名称。
    columns_to_extract = ['problem', 'answer']

    # --- 3. 指定列名重命名规则 ---
    # 这是一个字典（map），key是原始列名，value是新列名。
    # 如果你不想重命名任何列，请将此变量设置为空字典：{}
    column_rename_map = {
        'problem': 'ori_question',
        'answer': 'expected_answer',
    } 

    process_jsonl(input_file_path, output_file_path, columns_to_extract, column_rename_map)
