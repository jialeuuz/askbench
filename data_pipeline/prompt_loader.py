# prompt_loader.py
import re
from typing import Dict

def load_prompts(file_path: str) -> Dict[str, str]:
    """
    Load prompt templates from a file.

    The loader extracts all variables whose names start with "template_" and whose values are wrapped
    in triple quotes (''').

    Expected file format:
    template_name_1 = '''...content...'''
    template_name_2 = '''...content...'''
    """
    prompts = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Match "template_name = '''content'''" blocks
        pattern = r"(template_[\w]+)\s*=\s*'''(.*?)'''"
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            raise ValueError("No valid templates found. Expected format: `template_name = '''...'''`.")

        for name, template_content in matches:
            prompts[name.strip()] = template_content.strip()
            
        print(f"Loaded {len(prompts)} templates: {list(prompts.keys())}")
        return prompts

    except FileNotFoundError:
        print(f"Error: template file '{file_path}' not found.")
        raise
    except Exception as e:
        print(f"Error while loading templates: {e}")
        raise

# Example usage (assumes prompts.txt exists in the same directory)
if __name__ == '__main__':
    try:
        # Load your prompts.txt
        all_templates = load_prompts('prompts.txt')
        
        # Print one template for a quick sanity check
        # Make sure 'template_generate_ask_and_question' exists in your prompts file.
        if 'template_generate_ask_and_question' in all_templates:
            print("\n--- Sanity check: loaded template 'template_generate_ask_and_question' ---")
            print(all_templates['template_generate_ask_and_question'])
        else:
            print("'template_generate_ask_and_question' not found. Please check the template name.")

    except Exception as e:
        print(f"Failed to run the example: {e}")
