import json

# 输入输出路径
input_file = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/yitu/sample_medmcqa_2k_ask.jsonl"
output_file = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/data/2k_med_ask/train_jsonl/sample_medmcqa_2k_ask.jsonl"

all_data = []

# 读取原始数据
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        # 1. 保留原始完整数据
        all_data.append(data)
        
        # 2. 添加精简版数据
        simplified_data = {
            "ori_question": data["ori_question"],
            "expected_answer": data["expected_answer"]
        }
        all_data.append(simplified_data)

# 一次性写入
with open(output_file, 'w', encoding='utf-8') as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ 原始数据: {len(all_data)//2} 条（8列）")
print(f"✅ 精简数据: {len(all_data)//2} 条（2列）")
print(f"✅ 总计: {len(all_data)} 条")
print(f"✅ 已保存到 {output_file}")