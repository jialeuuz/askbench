import json
import os

def process_jsonl(input_file: str, output_file: str, columns_to_extract: list, column_rename_map: dict):
    """
    Process a JSONL file by extracting selected keys and optionally renaming them.

    Args:
        input_file: input JSONL file path
        output_file: output JSONL file path
        columns_to_extract: list of keys to keep from each JSON object
        column_rename_map: mapping {original_key: new_key}. If empty, no renaming is applied.
    """
    # Validate input
    if not os.path.exists(input_file):
        print(f"Error: input file '{input_file}' does not exist.")
        return

    print(f"Processing file: {input_file}")
    print(f"Extracting keys: {columns_to_extract}")
    if column_rename_map:
        print(f"Applying rename map: {column_rename_map}")
    else:
        print("No renaming will be applied.")

    # Use 'with' to ensure files are closed properly.
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Read line by line
        for line in infile:
            # Strip trailing whitespace/newlines
            line = line.strip()
            if not line:
                continue

            try:
                # Parse JSON
                data = json.loads(line)
                
                # Build a new record with extracted/renamed fields
                new_record = {}
                
                # Extract selected keys
                for original_key in columns_to_extract:
                    # Skip missing keys
                    if original_key in data:
                        # Determine output key name
                        new_key = column_rename_map.get(original_key, original_key)
                        
                        # Copy value over
                        new_record[new_key] = data[original_key]
                
                # Write if non-empty
                if new_record:
                    # ensure_ascii=False preserves non-ASCII characters
                    outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSON line -> {line}")

    print(f"Done. Output saved to: {output_file}")


if __name__ == '__main__':
    
    # --- 1) Input/output paths ---
    input_file_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/rubric/mindeval/data/common_language/math/aime2025/test.jsonl'
    output_file_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/ask_eval/data/common/aime2025/test_clear.jsonl'

    # --- 2) Keys to extract ---
    columns_to_extract = ['problem', 'answer']

    # --- 3) Optional rename map ---
    # If you don't want to rename any columns, set this to {}.
    column_rename_map = {
        'problem': 'ori_question',
        'answer': 'expected_answer',
    } 

    process_jsonl(input_file_path, output_file_path, columns_to_extract, column_rename_map)
