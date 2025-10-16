import json
import os

def convert_jsonl_to_training_format(input_file_path, output_file_path):
    """
    将包含 'conversation_history' 的 JSONL 文件转换为 LLaMA-Factory 所需的训练格式。

    Args:
        input_file_path (str): 输入的 JSONL 文件路径。
        output_file_path (str): 输出的 JSON 文件路径。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 用于存储所有转换后的对话
    all_formatted_conversations = []
    
    line_count = 0
    conversion_count = 0

    print(f"开始处理文件: {input_file_path}")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line_count += 1
                try:
                    # 1. 解析每一行的 JSON 数据
                    original_data = json.loads(line.strip())

                    # 2. 检查是否存在 'conversation_history' 键
                    if 'conversation_history' not in original_data:
                        print(f"警告: 第 {line_count} 行缺少 'conversation_history' 键，已跳过。")
                        continue

                    # 3. 提取 conversation_history
                    history = original_data['conversation_history']
                    
                    # 用于存储当前对话的转换后格式
                    transformed_turns = []

                    # 4. 遍历对话历史，进行格式转换
                    for turn in history:
                        role = turn.get('role')
                        content = turn.get('content')

                        if role is None or content is None:
                            print(f"警告: 第 {line_count} 行的某个对话轮次缺少 'role' 或 'content'，已跳过该轮次。")
                            continue

                        # 映射 role 到 from
                        if role == 'user':
                            from_key = 'human'
                        elif role == 'assistant':
                            from_key = 'gpt'
                        else:
                            print(f"警告: 第 {line_count} 行发现未知角色 '{role}'，已跳过该轮次。")
                            continue
                        
                        # 5. 创建新的对话轮次字典
                        new_turn = {
                            "from": from_key,
                            "value": content
                        }
                        transformed_turns.append(new_turn)
                    
                    # 6. 将转换后的对话轮次列表包装在最终结构中
                    if transformed_turns:
                        final_conversation_object = {
                            "conversations": transformed_turns
                        }
                        all_formatted_conversations.append(final_conversation_object)
                        conversion_count += 1

                except json.JSONDecodeError:
                    print(f"警告: 第 {line_count} 行不是有效的 JSON 格式，已跳过。")
                except Exception as e:
                    print(f"处理第 {line_count} 行时发生未知错误: {e}")

        # 7. 将所有数据写入到目标 JSON 文件
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # 使用 indent=2 使输出文件格式美观，易于阅读
            # ensure_ascii=False 确保中文字符能正确写入
            json.dump(all_formatted_conversations, outfile, ensure_ascii=False, indent=2)

        print("\n转换完成！")
        print(f"总共读取行数: {line_count}")
        print(f"成功转换对话数: {conversion_count}")
        print(f"输出文件已保存至: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到于 '{input_file_path}'")
    except Exception as e:
        print(f"发生严重错误: {e}")

if __name__ == '__main__':
    input_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/final_train_data/degrade_med_40k_oss120b_low.jsonl'
    output_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/models/train/LLaMA-Factory/data/degrade_med_40k_oss120b_low.json'
    convert_jsonl_to_training_format(input_path, output_path)