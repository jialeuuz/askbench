import json
import os

# Input file path
input_file = os.getenv("INPUT_FILE", "/path/to/input.jsonl")
# Output file path (write to a new file first to avoid corrupting the original)
output_file = os.getenv("OUTPUT_FILE", "/path/to/output.jsonl")

# Fixed suffix/prefix to add
# prefix_text = ("Please answer the following multiple-choice questions, ensuring your response concludes "
#                "with the correct option in the format: 'The answer is A.'.\n\n")
prefix_text = "\nPlease reason step by step, and put your final answer within \\boxed" + r"{}."

with open(input_file, 'r', encoding='utf-8') as fin, \
        open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        # Read JSON
        obj = json.loads(line)
        # If `ori_question` exists, append the suffix/prefix.
        if "ori_question" in obj:
            # obj["ori_question"] = prefix_text + obj["ori_question"]
            obj["ori_question"] = obj["ori_question"] + prefix_text
        # Write back
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"Done. Output saved to {output_file}")
