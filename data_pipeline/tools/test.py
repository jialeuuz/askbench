import json
import os

def convert_jsonl_to_training_format(input_file_path, output_file_path):
    """
    Converts a JSONL file with 'ori_question' and 'solution' to a
    JSON file with a 'conversations' list format for training.

    Args:
        input_file_path (str): The path to the source JSONL file.
        output_file_path (str): The path to the destination JSON file.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    converted_data = []
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    # Parse JSON object per line
                    original_record = json.loads(line)
                    
                    # Extract required fields
                    question = original_record.get("ori_question")
                    solution = original_record.get("solution")
                    
                    # Validate required fields
                    if question is None or solution is None:
                        print(f"Warning: Skipping line {i+1} due to missing 'ori_question' or 'solution'.")
                        continue
                        
                    # Build the training record format
                    new_record = {
                        "conversations": [
                            {
                                "from": "human",
                                "value": question
                            },
                            {
                                "from": "gpt",
                                "value": solution
                            }
                        ]
                    }
                    
                    converted_data.append(new_record)
                    
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {i+1} due to invalid JSON format.")
                except Exception as e:
                    print(f"An error occurred on line {i+1}: {e}")

        # Write converted data to output file
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # indent=2 for readability; ensure_ascii=False preserves non-ASCII characters
            json.dump(converted_data, outfile, ensure_ascii=False, indent=2)
            
        print(f"\nConversion successful!")
        print(f"Processed {len(converted_data)} records.")
        print(f"Input:  {input_file_path}")
        print(f"Output: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main ---
if __name__ == "__main__":
    # Set input/output paths
    input_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/useless/math_sample_20k.jsonl'
    output_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/models/train/LLaMA-Factory/data/math_sample_20k.json'
    
    # Run conversion
    convert_jsonl_to_training_format(input_path, output_path)
