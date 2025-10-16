# prompt_loader.py
import re
from typing import Dict

def load_prompts(file_path: str) -> Dict[str, str]:
    """
    从指定文件中加载所有以 "template_" 开头并用 ''' 包裹的prompt模板。
    文件格式应为:
    template_name_1 = '''...content...'''
    template_name_2 = '''...content...'''
    """
    prompts = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 正则表达式匹配 "template_name = '''content'''" 格式
        pattern = r"(template_[\w]+)\s*=\s*'''(.*?)'''"
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            raise ValueError("在文件中没有找到任何有效的模板。请检查格式是否为 `template_name = '''...'''`。")

        for name, template_content in matches:
            prompts[name.strip()] = template_content.strip()
            
        print(f"成功加载 {len(prompts)} 个模板: {list(prompts.keys())}")
        return prompts

    except FileNotFoundError:
        print(f"错误: 模板文件 '{file_path}' 未找到。")
        raise
    except Exception as e:
        print(f"加载模板时出错: {e}")
        raise

# 示例用法 (假设 prompts.txt 文件已存在于同一目录)
if __name__ == '__main__':
    try:
        # 直接加载您自己的 prompts.txt 文件
        all_templates = load_prompts('prompts.txt')
        
        # 打印其中一个加载的模板以作验证
        # 请确保 'template_generate_ask_and_question' 是您文件中的一个有效模板名
        if 'template_generate_ask_and_question' in all_templates:
            print("\n--- 验证加载的模板 'template_generate_ask_and_question' ---")
            print(all_templates['template_generate_ask_and_question'])
        else:
            print("\n'template_generate_ask_and_question' 未在文件中找到，请检查模板名称。")

    except Exception as e:
        print(f"无法执行示例: {e}")