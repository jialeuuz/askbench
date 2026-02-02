import json
import os

def convert_jsonl_to_training_format(input_file_path, output_file_path):
    """
    Convert a JSONL file containing `conversation_history` into the training JSON format expected by LLaMA-Factory.

    Args:
        input_file_path (str): input JSONL path.
        output_file_path (str): output JSON path.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Store converted conversations
    all_formatted_conversations = []
    
    line_count = 0
    conversion_count = 0

    print(f"Processing file: {input_file_path}")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line_count += 1
                try:
                    # 1) Parse one JSON line
                    original_data = json.loads(line.strip())

                    # 2) Validate key existence
                    if 'conversation_history' not in original_data:
                        print(f"Warning: line {line_count} missing 'conversation_history'; skipped.")
                        continue

                    # 3) Extract conversation_history
                    history = original_data['conversation_history']
                    
                    # Converted turns for this conversation
                    transformed_turns = []

                    # 4) Convert each turn
                    for turn in history:
                        role = turn.get('role')
                        content = turn.get('content')

                        if role is None or content is None:
                            print(f"Warning: line {line_count} has a turn missing 'role' or 'content'; skipped this turn.")
                            continue

                        # Map role -> "from"
                        if role == 'user':
                            from_key = 'human'
                        elif role == 'assistant':
                            from_key = 'gpt'
                        else:
                            print(f"Warning: line {line_count} has unknown role '{role}'; skipped this turn.")
                            continue
                        
                        # 5) Create new turn dict
                        new_turn = {
                            "from": from_key,
                            "value": content
                        }
                        transformed_turns.append(new_turn)
                    
                    # 6) Wrap into the final structure
                    if transformed_turns:
                        final_conversation_object = {
                            "conversations": transformed_turns
                        }
                        all_formatted_conversations.append(final_conversation_object)
                        conversion_count += 1

                except json.JSONDecodeError:
                    print(f"Warning: line {line_count} is not valid JSON; skipped.")
                except Exception as e:
                    print(f"Unexpected error while processing line {line_count}: {e}")

        # 7) Write output JSON
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # indent=2 makes the output easier to read.
            # ensure_ascii=False preserves non-ASCII characters.
            json.dump(all_formatted_conversations, outfile, ensure_ascii=False, indent=2)

        print("\nDone.")
        print(f"Total lines read: {line_count}")
        print(f"Conversations converted: {conversion_count}")
        print(f"Output written to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: input file not found: '{input_file_path}'")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == '__main__':
    input_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/final_train_data/degrade_med_40k_oss120b_low.jsonl'
    output_path = '/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/models/train/LLaMA-Factory/data/degrade_med_40k_oss120b_low.json'
    convert_jsonl_to_training_format(input_path, output_path)
