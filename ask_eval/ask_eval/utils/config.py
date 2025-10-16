# ask_eval/utils/config.py
import configparser
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import re
import datetime
from datetime import datetime

def load_config(config_path: str) -> configparser.ConfigParser:
    """加载配置文件"""
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    return config

def load_merged_config(base_config_path: str, task_config_path: str = None) -> configparser.ConfigParser:
    """加载并合并配置,基础配置优先级高于任务配置"""
    # 先加载任务配置
    config = configparser.ConfigParser()
    if task_config_path:
        config.read(task_config_path, encoding='utf-8')
    
    # 再加载基础配置,会覆盖任务配置中的同名项
    base_config = configparser.ConfigParser()
    base_config.read(base_config_path, encoding='utf-8')
    
    # 用基础配置覆盖任务配置
    for section in base_config.sections():
        if section == "evaluatorconfig":      # 评估器配置，如果任务配置中没有，则不添加
            if config.has_section(section):
                for key, value in base_config.items(section):
                    config.set(section, key, value)
            else:
                pass
        else:
            if not config.has_section(section):
                config.add_section(section)
            for key, value in base_config.items(section):
                config.set(section, key, value)
    
    return config

def get_section_config(config: configparser.ConfigParser, section: str, 
                     param_specs: Optional[Dict[str, Tuple[str, Any]]] = None) -> Dict:
    """获取指定部分的配置
    
    Args:
        config: 配置解析器对象
        section: 配置部分名称
        param_specs: 参数规格字典，格式为 {参数名: (类型, 默认值)}
                     类型可以是 'str', 'int', 'float', 'bool' 或 None(自动检测)
                     如果为None，将读取该部分的所有参数并自动检测类型
    
    Returns:
        包含该部分所有配置的字典
    """
    result = {}
    
    # 如果未指定部分，返回空字典
    if not config.has_section(section):
        return result
    
    # 如果指定了参数规格，按规格读取
    if param_specs:
        for param, (param_type, default) in param_specs.items():
            if param_type == 'int':
                result[param] = config.getint(section, param, fallback=default)
            elif param_type == 'float':
                result[param] = config.getfloat(section, param, fallback=default)
            elif param_type == 'bool':
                result[param] = config.getboolean(section, param, fallback=default)
            else:  # str或其他
                result[param] = config.get(section, param, fallback=default)
                if result[param] and result[param][0] == '"' and result[param][-1] == '"':
                    result[param] = result[param].strip('"')
    # 否则，读取所有参数并自动检测类型
    else:
        for key, value in config.items(section):
            # 尝试将数值转换为恰当的类型
            try:
                if "." in value and value.replace(".", "", 1).isdigit():
                    result[key] = float(value)
                elif value.isdigit():
                    result[key] = int(value)
                else:
                    result[key] = value
            except (ValueError, TypeError):
                result[key] = value
    
    return result

def get_model_config(config: configparser.ConfigParser) -> Dict:
    """获取模型配置"""
    param_specs = {
        "model_type": ("str", None),
        "api_type": ("str", "none"),
        "task_name": ("str", None),
        "sk_token": ("str", "none"),
        "api_url": ("str", None),
        "timeout": ("str", 600),
        "extra_prompt": ("str", None),
        "system_prompt": ("str", None),  # 新增system_prompt参数
        "enable_thinking": ("bool", True)  # 为qwen3新增enable_thinking参数
    }
    return get_section_config(config, "model", param_specs)

def get_generate_config(config: configparser.ConfigParser) -> Dict:
    """获取生成配置"""
    param_specs = {
        "max_tokens": ("int", 4096),
        "temperature": ("float", 0.6),
        "max_concurrent": ("int", 15),
        "shot": ("int", 0),
        "n_attempts": ("int", 1),  # 添加n_attempts参数，默认为1次尝试
        "top_k": ("int", -1),
        "top_p": ("float", -1)
    }
    return get_section_config(config, "generateconfig", param_specs)

def get_path_config(config: configparser.ConfigParser) -> Dict:
    """获取路径配置"""
    param_specs = {
        "data_dir": ("str", "data"),
        "save_dir": ("str", "results")
    }
    return get_section_config(config, "path", param_specs)

# def get_evaluator_config(config: configparser.ConfigParser) -> Dict:
#     """获取业务端提供的评估模型配置 或 调用GPT4o评估的模型配置 或 本地离线推理的模型路径"""
#     param_specs = {
#         "evaluator_url": ("str", None),
#         "headers_authorization": ("str", None),
#         "headers_content_type": ("str", None),
#         "max_concurrent": ("int", 1),
#         "time_out": ("int", 300),
#     }
#     return get_section_config(config, "evaluatorconfig", param_specs)
def get_evaluator_config(config: configparser.ConfigParser) -> Dict:
    """获取评估模型（裁判模型）的配置"""
    # 定义 create_model 函数需要的参数
    param_specs = {
        # --- 模型创建相关参数 ---
        "model_type": ("str", None),          # 必须！例如 'api'
        "api_type": ("str", None),            # 必须！例如 'deepseek', 'gpt-4o'
        "api_url": ("str", None),             # 必须！API 的 URL
        "sk_token": ("str", "none"),          # API 密钥
        "timeout": ("int", 600),              # 超时时间，注意键名是 'timeout'
        "system_prompt": ("str", None),       # 系统提示词
        
        # --- 文本生成相关参数 ---
        "temperature": ("float", 0.1),
        "max_new_tokens": ("int", 2048),
        "top_p": ("float", 1.0),
        
        # --- 其他控制参数 ---
        "max_concurrent": ("int", 10),
    }
    
    # 从 [evaluatorconfig] 部分读取配置
    eval_config = get_section_config(config, "evaluatorconfig", param_specs)

    # 兼容旧的 time_out 写法，如果存在就用它覆盖 timeout
    if config.has_option("evaluatorconfig", "time_out"):
        eval_config['timeout'] = config.getint("evaluatorconfig", "time_out")

    return eval_config

def get_charactereval_config(config: configparser.ConfigParser) -> Dict:
    """专为CharacterEval人设评估的配置文件读取"""
    param_specs = {
        "reward_model": ("str", "baichuan"),
        "reward_model_path": ("str", None),
        "max_seq_length": ("int", 4096),
    }
    return get_section_config(config, "characterevalconfig", param_specs)

def get_hallulensconfig_config(config: configparser.ConfigParser) -> Dict:
    """专为Hallulens幻觉评估的配置文件读取"""
    param_specs = {
        "do_generate_prompt": ("bool", True),
        "do_inference": ("bool", True),
        "do_eval": ("bool", True),
        "N": ("int", 5000),            # 生成和评估问题的数量
        "model_type": ("str", "api"),
        "api_url": ("str", None),      # 默认问题生成器和评估器是一样的
        "api_type": ("str", None),
        "max_tokens": ("int", 4096),
        "temperature": ("float", 0.6),
        "max_concurrent": ("int", 2),
        "time_out": ("int", 500)
    }
    return get_section_config(config, "hallulensconfig", param_specs)

def get_specific_config(config: configparser.ConfigParser, section_name: str) -> Dict:
    """获取特定评估器的配置（向后兼容）"""
    return get_section_config(config, section_name)

def write_final_result_file(save_dir: str, task: str, task_name: str, final_file_name: str = "final_result.txt") -> None:
    """将结果写入最终文件"""
    # 读取 save_dir/task/task_name 下的结果文件，获取最终精度
    task_result_path = os.path.join(save_dir, task, task_name, "results.txt")

    # 检查文件是否存在，如果不存在则直接 return
    if not os.path.exists(task_result_path):
        return
    
    with open(task_result_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 定义匹配模式：匹配准确率相关指标和总耗时
    pass_at_1_pattern = r'Pass@1.*?:\s*(\d+\.\d+)'
    pass_at_n_pattern = r'Pass@(\d+).*?:\s*(\d+\.\d+)'
    avg_at_n_pattern = r'Avg@(\d+).*?:\s*(\d+\.\d+)'
    legacy_acc_pattern = r'准确率:\s*(\d+\.\d+)'  # 兼容旧格式
    time_pattern = r"总耗时:\s*(\d+):(\d{2}):(\d{2}\.\d+)"
    
    # 使用集合来跟踪已添加的指标，避免重复
    added_metrics = set()
    metrics = []
    
    # 尝试匹配 Pass@1
    pass_at_1_match = re.search(pass_at_1_pattern, content)
    if pass_at_1_match:
        metrics.append(f"Pass@1: {pass_at_1_match.group(1)}")
        added_metrics.add("Pass@1")
    else:
        # 尝试匹配旧格式的准确率
        legacy_match = re.search(legacy_acc_pattern, content)
        if legacy_match:
            metrics.append(legacy_match.group(0))
        else:
            metrics.append("准确率: 未知")
    
    # 查找所有 Pass@n 匹配
    pass_at_n_matches = re.findall(pass_at_n_pattern, content)
    # 过滤掉 n=1 的情况，因为已经由 pass_at_1_pattern 处理
    for n, value in pass_at_n_matches:
        metric_key = f"Pass@{n}"
        if n != '1' and metric_key not in added_metrics:  # 避免重复
            metrics.append(f"{metric_key}: {value}")
            added_metrics.add(metric_key)
    
    # 尝试匹配 Avg@n
    avg_at_n_matches = re.findall(avg_at_n_pattern, content)
    for n, value in avg_at_n_matches:
        metric_key = f"Avg@{n}"
        if metric_key not in added_metrics:  # 避免重复
            metrics.append(f"{metric_key}: {value}")
            added_metrics.add(metric_key)
    
    # 查找总耗时
    time_match = re.search(time_pattern, content)
    if time_match:
        total_time = time_match.group(0)
    else:
        total_time = "总耗时: 未知"

    # 格式化结果
    metrics_str = " | ".join(metrics)
    
    final_file_path = os.path.join(save_dir, final_file_name)
    # 以追加模式打开最终文件
    with open(final_file_path, 'a', encoding='utf-8') as file:
        file.write(f"{task.ljust(30)} | {metrics_str.ljust(30)} | {total_time}\n")


def write_final_evalscope_result_file(
    save_dir: str,
    task: str,
    task_name: str,
    config: Dict[str, Any],  # 无默认值，必须传
    final_file_name: str = "final_result.txt",  # 默认参数放最后
) -> None:

    """将结果写入最终文件"""
    # 读取 save_dir/task/task_name 下的结果文件，获取最终精度
    tasks_config_dir = config.get("tasks", "tasks_config_path")
    if 'origin' in tasks_config_dir:
        task_result_path = os.path.join(save_dir, task, task_name, "results.txt")
        with open(task_result_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # 定义匹配模式：匹配准确率相关指标和总耗时
        pass_at_1_pattern = r'Pass@1.*?:\s*(\d+\.\d+)'
        pass_at_n_pattern = r'Pass@(\d+).*?:\s*(\d+\.\d+)'
        avg_at_n_pattern = r'Avg@(\d+).*?:\s*(\d+\.\d+)'
        score_pattern = r'score:\s*(\d+\.\d+)'
        legacy_acc_pattern = r'准确率:\s*(\d+\.\d+)'  # 兼容旧格式
        time_pattern = r"总耗时:\s*(\d+):(\d{2}):(\d{2}\.\d+)"
        
        # 使用集合来跟踪已添加的指标，避免重复
        added_metrics = set()
        metrics = []
        
        # 尝试匹配 Pass@1
        pass_at_1_match = re.search(pass_at_1_pattern, content)
        score_pattern_at_1_match = re.search(score_pattern, content)
        if pass_at_1_match:
            metrics.append(f"Pass@1: {pass_at_1_match.group(1)}")
            added_metrics.add("Pass@1")
        else:
            # 尝试匹配旧格式的准确率
            legacy_match = re.search(legacy_acc_pattern, content)
            if legacy_match:
                metrics.append(legacy_match.group(0))
            elif score_pattern_at_1_match:
                metrics.append(score_pattern_at_1_match.group(0))
            else:
                metrics.append("准确率: 未知")
        
        # 查找所有 Pass@n 匹配
        pass_at_n_matches = re.findall(pass_at_n_pattern, content)
        # 过滤掉 n=1 的情况，因为已经由 pass_at_1_pattern 处理
        for n, value in pass_at_n_matches:
            metric_key = f"Pass@{n}"
            if n != '1' and metric_key not in added_metrics:  # 避免重复
                metrics.append(f"{metric_key}: {value}")
                added_metrics.add(metric_key)
        
        # 尝试匹配 Avg@n
        avg_at_n_matches = re.findall(avg_at_n_pattern, content)
        for n, value in avg_at_n_matches:
            metric_key = f"Avg@{n}"
            if metric_key not in added_metrics:  # 避免重复
                metrics.append(f"{metric_key}: {value}")
                added_metrics.add(metric_key)
        
        # 查找总耗时
        time_match = re.search(time_pattern, content)
        if time_match:
            total_time = time_match.group(0)
        else:
            total_time = "总耗时: 未知"

        # 格式化结果
        metrics_str = " | ".join(metrics)
        
        final_file_path = os.path.join(save_dir, final_file_name)
        # 以追加模式打开最终文件
        with open(final_file_path, 'a', encoding='utf-8') as file:
            file.write(f"{task.ljust(30)} | {metrics_str.ljust(30)} | {total_time}\n")

    elif 'OpenCompass' in tasks_config_dir:
        date_pattern = re.compile(r'^\d{8}_\d{6}$')
    
        # 获取所有符合时间格式的文件夹
        time_dirs = []
        task_dir = os.path.join(save_dir, task, task_name)
        
        if not os.path.exists(task_dir):
            print(f"任务目录不存在: {task_dir}")
            return
        
        for dir_name in os.listdir(task_dir):
            dir_path = os.path.join(task_dir, dir_name)
            if os.path.isdir(dir_path) and date_pattern.match(dir_name):
                try:
                    # 将目录名转换为datetime对象以便排序
                    dir_time = datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
                    time_dirs.append((dir_time, dir_name, dir_path))
                except ValueError:
                    continue
        
        if not time_dirs:
            print(f"在 {task_dir} 中未找到符合时间格式的目录")
            return
        
        # 按时间降序排序，获取最新的目录
        time_dirs.sort(reverse=True)
        latest_time, latest_dir_name, latest_dir_path = time_dirs[0]
        
        # 构建结果文件路径
        task_result_path = os.path.join(latest_dir_path, "results.txt")
        
        if not os.path.exists(task_result_path):
            print(f"在最新目录 {latest_dir_name} 中未找到 results.txt")
            return


        with open(task_result_path, 'r', encoding='utf-8') as file:
            content = file.read()

         # 定义匹配模式：匹配准确率相关指标和总耗时
        pass_at_1_pattern = r'Pass@1.*?:\s*(\d+\.\d+)'
        pass_at_n_pattern = r'Pass@(\d+).*?:\s*(\d+\.\d+)'
        avg_at_n_pattern = r'Avg@(\d+).*?:\s*(\d+\.\d+)'
        score_pattern = r'score:\s*(\d+\.\d+)'
        legacy_acc_pattern = r'准确率:\s*(\d+\.\d+)'  # 兼容旧格式
        time_pattern = r"总耗时:\s*(\d+):(\d{2}):(\d{2}\.\d+)"
        # —— 新增处理多文件格式开始 ——
        # 判断是多文件（多个“评估文件:”或“文件名:”出现）
        folder_line = ""
        folder_match = re.search(r'(?:文件名|文件夹):\s*\d+_\d+', content)
        if folder_match:
            folder_line = folder_match.group(0)

        multi_file = len(re.findall(r'(评估文件:|文件名:)', content)) > 1

        if multi_file:
            blocks = re.split(r'(?:评估文件:|文件名:)', content)[1:]  # 跳过前言
            metrics = []
            time_match = re.search(time_pattern, content)
            total_time = time_match.group(0) if time_match else "总耗时: 未知"
            for block in blocks:
                b_head = block.strip()
                fname_match = re.match(r"([^\n\r]+)", b_head)
                if fname_match:
                    fname = fname_match.group(1).replace('.json', '').replace('-', '.')
                else:
                    fname = "未知文件"
                # 准确率/score
                m_acc = re.search(r'准确率:\s*(\d+\.\d+)', block)
                m_score = re.search(r'score:\s*(\d+\.\d+)', block)
                if m_acc:
                    acc = f"准确率: {m_acc.group(1)}"
                elif m_score:
                    acc = f"score: {m_score.group(1)}"
                else:
                    acc = "准确率: 未知"
                metrics.append(f"{fname} | {acc}")

            metrics_str = "   ".join(metrics)
            # 拼接格式：任务名 | 文件名: 20250430_101042 | bustm.test | 准确率: 80.00 ... | 总耗时: XXX
            final_file_path = os.path.join(save_dir, final_file_name)
            with open(final_file_path, 'a', encoding='utf-8') as file:
                file.write(f"{task.ljust(20)} | {folder_line} | {metrics_str} | {total_time}\n")

        else:
            legacy_match = re.search(legacy_acc_pattern, content)
            score_match = re.search(score_pattern, content)
            pass1_match = re.search(pass_at_1_pattern, content)
            if pass1_match:
                acc_str = f"Pass@1: {pass1_match.group(1)}"
            elif legacy_match:
                acc_str = legacy_match.group(0)
            elif score_match:
                acc_str = score_match.group(0)
            else:
                acc_str = "准确率: 未知"
            time_match = re.search(time_pattern, content)
            total_time = time_match.group(0) if time_match else "总耗时: 未知"
            metrics_str = " | ".join(x for x in [folder_line, acc_str] if x)
            final_file_path = os.path.join(save_dir, final_file_name)
            with open(final_file_path, 'a', encoding='utf-8') as file:
                file.write(f"{task.ljust(30)} | {metrics_str} | {total_time}\n")

    elif 'VLMEvalKit' in tasks_config_dir:
        api_type = config.get("model", "api_type")
        task_result_path = os.path.join(save_dir, task, task_name, api_type, "results.txt")
        
        try:
            with open(task_result_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
        except FileNotFoundError:
            print(f"结果文件未找到: {task_result_path}")
            return
        
        # 初始化变量
        metrics = []
        time_info = "总耗时: 未知"
        all_metrics = []
        collect_all = False
        
        # 定义匹配模式
        patterns = {
            'pass_at_1': r'Pass@1.*?:\s*(\d+\.\d+)',
            'pass_at_n': r'Pass@(\d+).*?:\s*(\d+\.\d+)',
            'avg_at_n': r'Avg@(\d+).*?:\s*(\d+\.\d+)',
            'overall': r'Overall:\s*(\d+\.\d+)',
            'score': r'score:\s*(\d+\.\d+)',
            'legacy_acc': r'准确率:\s*(\d+\.\d+)',
            'time': r"总耗时:\s*(\d+:\d{2}:\d{2}\.\d+)",
            'metric_line': r'^([^:]+):\s*([^\n]+)$'  # 匹配所有指标行
        }
        
        # 检查是否有Overall指标
        has_overall = any(re.search(patterns['overall'], line) for line in content)
        
        for line in content:
            line = line.strip()
            
            # 匹配总耗时
            time_match = re.search(patterns['time'], line)
            if time_match:
                time_info = f"总耗时: {time_match.group(1)}"
                continue
            
            # 如果有Overall指标，按原逻辑处理
            if has_overall:
                # 匹配Pass@1
                pass_at_1_match = re.search(patterns['pass_at_1'], line)
                if pass_at_1_match and 'Pass@1' not in [m.split(':')[0] for m in metrics]:
                    metrics.append(f"Pass@1: {pass_at_1_match.group(1)}")
                    continue
                    
                # 匹配旧格式准确率
                legacy_match = re.search(patterns['legacy_acc'], line)
                if legacy_match and '准确率' not in [m.split(':')[0] for m in metrics]:
                    metrics.append(legacy_match.group(0))
                    continue
                    
                # 匹配score
                score_match = re.search(patterns['score'], line)
                if score_match and 'score' not in [m.split(':')[0] for m in metrics]:
                    metrics.append(score_match.group(0))
                    continue
                    
                # 匹配Overall
                overall_match = re.search(patterns['overall'], line)
                if overall_match and 'Overall' not in [m.split(':')[0] for m in metrics]:
                    metrics.append(f"准确率: {overall_match.group(1)}")
                    continue
                    
                # 匹配Pass@n和Avg@n
                pass_at_n_match = re.search(patterns['pass_at_n'], line)
                if pass_at_n_match:
                    metric = f"Pass@{pass_at_n_match.group(1)}: {pass_at_n_match.group(2)}"
                    if metric.split(':')[0] not in [m.split(':')[0] for m in metrics]:
                        metrics.append(metric)
                        continue
                        
                avg_at_n_match = re.search(patterns['avg_at_n'], line)
                if avg_at_n_match:
                    metric = f"Avg@{avg_at_n_match.group(1)}: {avg_at_n_match.group(2)}"
                    if metric.split(':')[0] not in [m.split(':')[0] for m in metrics]:
                        metrics.append(metric)
                        continue
            else:
                # 如果没有Overall指标，收集所有指标行
                metric_match = re.search(patterns['metric_line'], line)
                if metric_match and line not in ['评估集', '开始时间']:
                    metric_name = metric_match.group(1).strip()
                    metric_value = metric_match.group(2).strip()
                    if metric_name not in [m.split(':')[0] for m in all_metrics]:
                        all_metrics.append(f"{metric_name}: {metric_value}")
        
        # 决定使用哪些指标
        if has_overall and metrics:
            metrics_str = " | ".join(metrics)
        elif all_metrics:
            metrics_str = " | ".join(all_metrics)
        else:
            metrics_str = "无有效指标"
        
        # 写入最终结果文件
        final_file_path = os.path.join(save_dir, final_file_name)
        with open(final_file_path, 'a', encoding='utf-8') as file:
            file.write(f"{task.ljust(30)} | {metrics_str.ljust(60)} | {time_info}\n")