import json

# 输入文件路径
input_file = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/ask_eval/data/common/aime2025/test_clear.jsonl"
# 输出文件路径（建议先写到新文件以免损坏原文件）
output_file = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/ask_eval/data/common/aime2025/test.jsonl"

# 要添加的固定前缀
# prefix_text = ("Please answer the following multiple-choice questions, ensuring your response concludes "
#                "with the correct option in the format: 'The answer is A.'.\n\n")
prefix_text = "\nPlease reason step by step, and put your final answer within \\boxed" + r"{}."

with open(input_file, 'r', encoding='utf-8') as fin, \
        open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        # 读取 json
        obj = json.loads(line)
        # 如果存在 ori_question 列，加上前缀
        if "ori_question" in obj:
            # obj["ori_question"] = prefix_text + obj["ori_question"]
            obj["ori_question"] = obj["ori_question"] + prefix_text
        # 写回到新文件
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"修改完成，结果保存在 {output_file}")