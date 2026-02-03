# -*- coding: utf-8 -*-
import json
import os

def convert_expected_answer_to_string(input_file_path, output_file_path):
    """
    读取一个 JSONL 文件，将其 'expected_answer' 字段的值转换为字符串，
    并保存到新的 JSONL 文件中。

    :param input_file_path: 输入的 JSONL 文件路径。
    :param output_file_path: 输出的 JSONL 文件路径。
    """
    print(f"开始处理文件: {input_file_path}")
    
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 打开输入文件进行读取，打开输出文件进行写入
        # 使用 'utf-8' 编码以正确处理可能存在的中文字符
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            # 逐行处理输入文件
            for i, line in enumerate(infile, 1):
                # 去除行首行尾的空白字符
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                try:
                    # 将行内容解析为 JSON 对象（Python 字典）
                    data = json.loads(line)

                    # 检查 'expected_answer' 键是否存在
                    if 'expected_answer' in data:
                        # 将 'expected_answer' 键对应的值强制转换为字符串类型
                        data['expected_answer'] = str(data['expected_answer'])

                    # 将修改后的字典转换回 JSON 字符串，并写入输出文件
                    # ensure_ascii=False 确保中文字符能被正确写入，而不是被转义成 ASCII 码
                    # 在末尾添加换行符，以保持 JSONL (JSON Lines) 格式
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

                except json.JSONDecodeError:
                    # 如果某一行不是有效的 JSON 格式，打印警告信息
                    print(f"警告: 第 {i} 行不是有效的 JSON 格式，已跳过。内容: {line}")
                except KeyError:
                    # 如果 'expected_answer' 键不存在，打印警告
                    print(f"警告: 第 {i} 行缺少 'expected_answer' 键，已原样写入。")
                    outfile.write(line + '\n')


        print(f"\n处理完成！")
        print(f"总共处理了 {i} 行。")
        print(f"修改后的数据已保存到: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 主程序入口 ---
if __name__ == '__main__':
    # 定义输入和输出文件的路径
    input_path = os.getenv("INPUT_FILE", "/path/to/input.jsonl")
    
    # 建议将结果写入一个新文件，以避免覆盖原始数据
    # 新文件名中加入了 "_string_answers" 以作区分
    output_path = os.getenv("OUTPUT_FILE", "/path/to/output.jsonl")
    
    # 调用函数执行转换
    convert_expected_answer_to_string(input_path, output_path)
