# verl/utils/reward_score/math_hard.py
import os
import random
import time
import math
import re
from typing import List, Tuple, Optional, Dict, Any
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 从环境变量获取API URLs，如果没有则使用默认值
API_URLS = [
    "http://10.72.0.14:8001/v1/chat/completions",
    "http://10.72.0.14:8002/v1/chat/completions",
    "http://10.72.0.14:8003/v1/chat/completions",
    "http://10.72.0.14:8004/v1/chat/completions"
]
# .env 文件路径
ENV_FILE_PATH = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhouyang/verl_v5/.env"


def load_api_urls_from_env(env_path: str = ENV_FILE_PATH, max_retries: int = 3, retry_delay: float = 1.0) -> List[str]:
    """从.env文件中读取并解析API URLs，支持重试机制"""
    
    for attempt in range(max_retries):
        try:
            if not os.path.exists(env_path):
                return API_URLS
            
            with open(env_path, 'r') as f:
                content = f.read()
            
            # 使用正则表达式匹配多行的 VLLM_BASE_URL
            pattern = r'VLLM_BASE_URL\s*=\s*["\']([^"\']*(?:\n[^"\']*)*)["\']'
            match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            
            if match:
                url_content = match.group(1).strip()
                
                # 移除所有换行符和多余空格
                url_content = re.sub(r'\s+', ' ', url_content)
                
                # 分割并清理URLs
                urls = [url.strip() for url in url_content.split(',') if url.strip()]
                
                # 拼接完整的API endpoint
                complete_urls = []
                for url in urls:
                    if url:
                        url = url.strip()
                        # 确保URL格式正确
                        if url.endswith('/v1'):
                            url = url + '/chat/completions'
                        elif not url.endswith('/chat/completions'):
                            url = url.rstrip('/') + '/v1/chat/completions'
                        complete_urls.append(url)
                
                if complete_urls:
                    return complete_urls
            
            return API_URLS
            
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                env_path = os.path.abspath(env_path)
                continue
            else:
                import traceback
                traceback.print_exc()
                return API_URLS
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return API_URLS
    
    return API_URLS


def extract_json_from_response(response_text: str) -> Optional[Dict[str, str]]:
    """
    从响应文本中提取JSON，支持多种格式
    
    Args:
        response_text: API返回的文本
        
    Returns:
        提取的JSON字典，包含reasoning和result字段；失败返回None
    """
    # 尝试1: 直接解析整个响应为JSON
    try:
        data = json.loads(response_text.strip())
        if 'reasoning' in data and 'result' in data:
            return data
    except json.JSONDecodeError:
        pass
    
    # 尝试2: 查找被代码块包裹的JSON
    json_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_block_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if 'reasoning' in data and 'result' in data:
                    return data
            except json.JSONDecodeError:
                continue
    
    # 尝试3: 查找任何JSON对象
    json_pattern = r'\{[^{}]*"reasoning"[^{}]*"result"[^{}]*\}'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if 'reasoning' in data and 'result' in data:
                return data
        except json.JSONDecodeError:
            pass
    
    # 尝试4: 分别提取reasoning和result字段
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"([^"]*)"',
        r"'reasoning'\s*:\s*'([^']*)'",
        r'reasoning:\s*([^\n]+)',
    ]
    
    result_patterns = [
        r'"result"\s*:\s*"(CORRECT|INCORRECT|CLARIFICATION)"',
        r"'result'\s*:\s*'(CORRECT|INCORRECT|CLARIFICATION)'",
        r'result:\s*(CORRECT|INCORRECT|CLARIFICATION)',
    ]
    
    reasoning = None
    result = None
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            break
    
    for pattern in result_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            result = match.group(1).strip().upper()
            break
    
    if reasoning and result:
        return {'reasoning': reasoning, 'result': result}
    
    return None


def extract_reasoning_quality_json(response_text: str) -> Optional[Dict[str, Any]]:
    """
    改进的JSON提取，处理多行字符串
    """
    import json
    
    # 尝试1: 直接解析
    try:
        data = json.loads(response_text.strip())
        if 'reasoning' in data and 'has_unauthorized_assumption' in data and 'reversal_count' in data:
            return data
    except json.JSONDecodeError:
        pass
    
    # 尝试2: 提取代码块中的JSON
    json_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_block_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if 'reasoning' in data and 'has_unauthorized_assumption' in data and 'reversal_count' in data:
                    return data
            except json.JSONDecodeError:
                continue
    
    # 尝试3: 查找第一个完整的JSON对象（改进的正则）
    # 使用更宽松的匹配，允许多行
    json_pattern = r'\{[^{}]*?"reasoning"[^{}]*?"has_unauthorized_assumption"[^{}]*?"reversal_count"[^{}]*?\}'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        try:
            json_str = match.group(0)
            data = json.loads(json_str)
            if 'reasoning' in data and 'has_unauthorized_assumption' in data and 'reversal_count' in data:
                return data
        except json.JSONDecodeError:
            pass
    
    # 尝试4: 手动清理并解析
    # 移除response_text前后的非JSON内容
    try:
        # 找到第一个 { 和最后一个 }
        start = response_text.find('{')
        end = response_text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
            
            # 尝试解析
            try:
                data = json.loads(json_str)
                if 'reasoning' in data and 'has_unauthorized_assumption' in data and 'reversal_count' in data:
                    return data
            except json.JSONDecodeError:
                # 尝试修复常见问题
                # 1. 替换单引号为双引号
                json_str_fixed = json_str.replace("'", '"')
                
                # 2. 修复布尔值
                json_str_fixed = re.sub(r'\bTrue\b', 'true', json_str_fixed)
                json_str_fixed = re.sub(r'\bFalse\b', 'false', json_str_fixed)
                
                try:
                    data = json.loads(json_str_fixed)
                    if 'reasoning' in data and 'has_unauthorized_assumption' in data and 'reversal_count' in data:
                        return data
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    
    # 尝试5: 分别提取字段（最后的备用方案）
    try:
        # 提取reasoning（允许多行，使用非贪婪匹配）
        reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text, re.DOTALL)
        
        # 提取has_unauthorized_assumption
        assumption_match = re.search(r'"has_unauthorized_assumption"\s*:\s*(true|false)', response_text, re.IGNORECASE)
        
        # 提取reversal_count
        reversal_match = re.search(r'"reversal_count"\s*:\s*(\d+)', response_text)
        
        if reasoning_match and assumption_match and reversal_match:
            return {
                'reasoning': reasoning_match.group(1).strip(),
                'has_unauthorized_assumption': assumption_match.group(1).lower() == 'true',
                'reversal_count': int(reversal_match.group(1))
            }
    except Exception as e:
        print(f"Field extraction error: {e}")
    
    return None


def call_llm_api(prompt: str, max_retries: int = 10) -> Dict[str, str]:
    """
    调用LLM API进行答案正确性判断，每次失败时重新加载API列表
    
    Returns:
        {"reasoning": str, "result": str} 或 {"error": "ERROR"}
    """
    
    api_urls = load_api_urls_from_env()
    
    for attempt in range(max_retries):
        try:
            # 每次重试时重新加载API列表
            if attempt > 0:
                api_urls = load_api_urls_from_env()
            
            if not api_urls:
                print(f"Attempt {attempt + 1}: No API URLs available")
                if attempt == max_retries - 1:
                    return {"error": "ERROR"}
                time.sleep(0.5)
                continue
            
            # 随机选择一个API
            api_url = random.choice(api_urls)
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "grader",
                "messages": [
                    {"role": "system", "content": "You are a precise mathematical answer evaluator. You must return valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 16000
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 尝试提取JSON
                extracted = extract_json_from_response(content)
                
                if extracted and 'result' in extracted:
                    # 验证result字段的值
                    if extracted['result'] in ['CORRECT', 'INCORRECT', 'CLARIFICATION']:
                        return extracted
                    else:
                        print(f"Invalid result value: {extracted['result']}, retrying...")
                else:
                    print(f"Failed to extract valid JSON from response, attempt {attempt + 1}/{max_retries}")
                    print(f"Response content: {content[:200]}")
                    # 继续重试
                
            else:
                if attempt == max_retries - 1:
                    print(f"API call failed with status {response.status_code}")
                    print(f"Response content: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}")
            time.sleep(0.5)
        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:
                print(f"Connection error on attempt {attempt + 1}: {str(e)}")
            time.sleep(0.5)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            time.sleep(0.5)
    
    print(f"All {max_retries} attempts failed, using fallback logic")
    return {"error": "ERROR"}


def call_reasoning_quality_api(prompt: str, max_retries: int = 10) -> Dict[str, Any]:
    """
    调用LLM API进行推理质量判断，每次失败时重新加载API列表
    
    Returns:
        {"reasoning": str, "has_unauthorized_assumption": bool, "reversal_count": int} 或 {"error": "ERROR"}
    """
    
    api_urls = load_api_urls_from_env()
    
    for attempt in range(max_retries):
        try:
            # 每次重试时重新加载API列表
            if attempt > 0:
                api_urls = load_api_urls_from_env()
            
            if not api_urls:
                print(f"Reasoning quality attempt {attempt + 1}: No API URLs available")
                if attempt == max_retries - 1:
                    return {"error": "ERROR"}
                time.sleep(0.5)
                continue
            
            # 随机选择一个API
            api_url = random.choice(api_urls)
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "grader",
                "messages": [
                    {"role": "system", "content": "You are a precise reasoning quality evaluator. You must return valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 16000
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 尝试提取JSON
                extracted = extract_reasoning_quality_json(content)
                
                if extracted:
                    # 验证字段类型
                    if isinstance(extracted.get('has_unauthorized_assumption'), bool) and \
                       isinstance(extracted.get('reversal_count'), int) and \
                       extracted.get('reversal_count', -1) >= 0:
                        return extracted
                    else:
                        print(f"Invalid field types in response, retrying...")
                else:
                    print(f"Failed to extract valid JSON from reasoning quality response, attempt {attempt + 1}/{max_retries}")
                    print(f"Response content: {content[:1200]}")
                
            else:
                if attempt == max_retries - 1:
                    print(f"Reasoning quality API call failed with status {response.status_code}")
                    print(f"Response content: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                print(f"Reasoning quality timeout on attempt {attempt + 1}")
            time.sleep(0.5)
        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:
                print(f"Reasoning quality connection error on attempt {attempt + 1}: {str(e)}")
            time.sleep(0.5)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Reasoning quality unexpected error on attempt {attempt + 1}: {str(e)}")
            time.sleep(0.5)
    
    print(f"All {max_retries} reasoning quality attempts failed")
    return {"error": "ERROR"}


def fallback_answer_check(solution_str: str, ground_truth: str) -> str:
    """
    当API不可用时的备用答案检查逻辑
    
    Args:
        solution_str: 学生的解答
        ground_truth: 标准答案
    
    Returns:
        "CORRECT", "INCORRECT", or "CLARIFICATION"
    """
    solution_lower = solution_str.lower()
    solution_stripped = solution_str.strip()
    
    # ========== 1. 检查是否是澄清/无法解决 ==========
    
    # 强澄清关键词（直接承认不会）
    strong_clarification_patterns = [
        r"\b(cannot|can't|unable to|don't know how to)\s+(solve|answer|figure out)",
        r"\b(i|we)\s+(need|require)\s+(help|clarification|more information)",
        r"(beyond my|outside my)\s+(capability|ability|knowledge)",
        r"(too|very)\s+(difficult|complex|hard)\s+(for me|to solve)",
        r"\b(i'm not|i am not)\s+(sure|certain|confident)",
        r"(could you|can you|please)\s+(clarify|explain|help)",
        r"(not clear|unclear|ambiguous|confusing)",
        r"(don't understand|do not understand|didn't understand)",
        r"(i apologize|sorry).+(cannot|can't|unable)",
    ]
    
    for pattern in strong_clarification_patterns:
        if re.search(pattern, solution_lower):
            return "CLARIFICATION"
    
    # 弱澄清信号（没有实质性答案）
    if len(solution_stripped) < 50:  # 回答很短
        # 检查是否包含疑问句但没有数字答案
        if '?' in solution_str and not re.search(r'\d+', solution_str):
            return "CLARIFICATION"
        
        # 只包含简短的不确定表达
        weak_patterns = [
            r'^(hmm|well|uh|um)',
            r'^(i think|maybe|perhaps|possibly)',
            r'(not sure|unsure)',
        ]
        for pattern in weak_patterns:
            if re.search(pattern, solution_lower):
                # 如果没有明确的数字答案，判定为澄清
                if not re.search(r'\d+\.?\d*', solution_str):
                    return "CLARIFICATION"
    
    # 检查是否只是复述问题而没有给出答案
    if solution_str.count('?') >= 2 and not re.search(r'(answer|solution|result)\s*is', solution_lower):
        return "CLARIFICATION"
    
    # ========== 2. 检查答案正确性 ==========
    
    try:
        # 提取 ground truth 中的数字
        gt_numbers = re.findall(r'-?\d+\.?\d*', ground_truth)
        
        # 提取 solution 中的数字
        sol_numbers = re.findall(r'-?\d+\.?\d*', solution_str)
        
        if not gt_numbers:
            # ground truth 没有数字，尝试文本匹配
            gt_normalized = re.sub(r'[^\w\s]', '', ground_truth.lower()).strip()
            sol_normalized = re.sub(r'[^\w\s]', '', solution_lower).strip()
            
            if gt_normalized in sol_normalized or sol_normalized in gt_normalized:
                return "CORRECT"
            
            if re.search(r'(answer|solution|result)\s+is', solution_lower):
                return "INCORRECT"
            
            return "INCORRECT"
        
        if not sol_numbers:
            return "INCORRECT"
        
        # 比较数字答案
        gt_num = float(gt_numbers[-1])
        
        for sol_num_str in reversed(sol_numbers):
            try:
                sol_num = float(sol_num_str)
                
                if abs(gt_num - sol_num) < 0.0001:
                    return "CORRECT"
                
                if abs(gt_num * 100 - sol_num) < 0.0001 or abs(gt_num - sol_num * 100) < 0.0001:
                    return "CORRECT"
                
                if sol_num != 0 and abs(1.0 / sol_num - gt_num) < 0.0001:
                    return "CORRECT"
                if gt_num != 0 and abs(1.0 / gt_num - sol_num) < 0.0001:
                    return "CORRECT"
                
            except (ValueError, ZeroDivisionError):
                continue
        
        return "INCORRECT"
        
    except Exception as e:
        print(f"Error in fallback answer check: {e}")
        return "INCORRECT"


def fallback_reasoning_quality_check(ori_question: str, solution_str: str) -> Dict[str, Any]:
    """
    当API不可用时的备用推理质量检查
    
    Args:
        ori_question: 原始问题
        solution_str: 学生的解答
    
    Returns:
        {"has_unauthorized_assumption": bool, "reversal_count": int}
    """
    solution_lower = solution_str.lower()
    
    # 检查未经授权的假设关键词
    assumption_keywords = [
        r'\b(let\'s assume|let us assume|assuming|suppose|let\'s say)\b',
        r'\b(假设|假定|设|令)\b',
    ]
    
    has_assumption = False
    for pattern in assumption_keywords:
        if re.search(pattern, solution_lower):
            has_assumption = True
            break
    
    # 检查转折词
    reversal_patterns = [
        r'\bwait\b',
        r'\bhold on\b',
        r'\bactually\b',
        r'\bon second thought\b',
        r'\blet me reconsider\b',
        r'\bthat\'s wrong\b',
        r'\bthat\'s incorrect\b',
        r'\bi need to rethink\b',
        r'不对',
        r'不行',
    ]
    
    reversal_count = 0
    for pattern in reversal_patterns:
        matches = re.findall(pattern, solution_lower)
        reversal_count += len(matches)
    
    return {
        "has_unauthorized_assumption": has_assumption,
        "reversal_count": reversal_count
    }


def check_answer_correctness(solution_str: str, ground_truth: str) -> str:
    """
    判断答案是否正确（第一个judge）
    
    Args:
        solution_str: 学生的解答
        ground_truth: 标准答案
    
    Returns:
        "CORRECT", "INCORRECT", or "CLARIFICATION"
    """
    
    # 构建prompt
    prompt = f"""You are evaluating a mathematical solution. Compare the student's answer with the correct answer.

Student's Solution:
{solution_str}

Correct Answer:
{ground_truth}

You MUST respond in valid JSON format with the following structure:
{{
    "reasoning": "Your detailed explanation here",
    "result": "CORRECT|INCORRECT|CLARIFICATION"
}}

Instructions:
1. First, extract the FINAL NUMERICAL ANSWER from the student's solution (ignore the reasoning process)
2. Compare it with the correct answer NUMERICALLY (not textually)
3. Determine the result:
   - CORRECT: The final numerical answer matches the correct answer (values must be mathematically equal, e.g., 0.5 = 1/2 = 50%)
   - INCORRECT: The final numerical answer does NOT match the correct answer
   - CLARIFICATION: The student explicitly states they cannot solve the problem or need help (e.g., "I cannot solve this", "I need clarification", "This is beyond my capability")

CRITICAL RULES:
- Focus ONLY on the final numerical answer, NOT the solution process
- Compare numerical VALUES, not text format
- "Answer: 18" when correct answer is 270 → INCORRECT
- "Answer: 0.5" when correct answer is 50% → CORRECT (same value)
- Only return CLARIFICATION if student explicitly admits inability to solve

Your response MUST be valid JSON. Do not include any text outside the JSON object.

Example response:
{{
    "reasoning": "The student's final answer is 18, but the correct answer is 270. These are different values.",
    "result": "INCORRECT"
}}"""
    
    # 调用API
    result = call_llm_api(prompt)
    
    # 如果API调用失败，使用备用逻辑
    if "error" in result:
        print("Answer correctness API call failed, using fallback logic")
        return fallback_answer_check(solution_str, ground_truth)
    
    # 解析result字段
    answer_status = result.get('result', '').strip().upper()
    
    # 验证返回值
    if answer_status in ['CORRECT', 'INCORRECT', 'CLARIFICATION']:
        return answer_status
    else:
        # 如果返回值无效，使用fallback
        print(f"Invalid answer status from API: {answer_status}, using fallback")
        return fallback_answer_check(solution_str, ground_truth)


def check_reasoning_quality(ori_question: str, solution_str: str) -> Dict[str, Any]:
    """
    判断推理质量（第二个judge）
    """
    
    # 改进的prompt
    prompt = f"""You are evaluating the reasoning quality of a mathematical solution. 

Original Question:
{ori_question}

Student's Solution:
{solution_str}

You MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.

CRITICAL JSON FORMAT REQUIREMENTS:
1. The "reasoning" field must be a SINGLE LINE string (no line breaks, use \\n for newlines if needed)
2. Escape all special characters properly
3. Use double quotes for strings
4. Boolean values must be lowercase: true/false (not True/False)
5. Numbers must be integers without quotes

Required JSON structure:
{{
    "reasoning": "Your analysis in a single line",
    "has_unauthorized_assumption": true,
    "reversal_count": 2
}}

Evaluation criteria:

1. UNAUTHORIZED ASSUMPTIONS (has_unauthorized_assumption):
   - Set to true if the student assumes information NOT provided in the question
   - Examples: "Let's assume X=10", "Assuming it's a square", "Let's say the speed is constant"
   - Set to false if all assumptions are justified

2. REASONING REVERSALS (reversal_count):
   - Count phrases like: "wait", "actually", "hold on", "不对", "let me reconsider", "that's wrong"
   - Return the total count as an integer

IMPORTANT: Your response must be ONLY the JSON object, nothing else. Keep the reasoning field concise (under 500 characters).

Example valid response:
{{"reasoning": "The student assumed AB is horizontal (line 5) without justification, which is unauthorized. Found 2 reversals: 'wait' at line 8 and 'actually' at line 12.", "has_unauthorized_assumption": true, "reversal_count": 2}}"""
    
    # 调用API
    result = call_reasoning_quality_api(prompt)
    
    # 如果API调用失败，使用备用逻辑
    if "error" in result:
        print("Reasoning quality API call failed, using fallback logic")
        return fallback_reasoning_quality_check(ori_question, solution_str)
    
    return {
        "has_unauthorized_assumption": result.get('has_unauthorized_assumption', False),
        "reversal_count": result.get('reversal_count', 0)
    }


def calculate_base_reward(pass_rate: float, answer_status: str) -> float:
    """
    根据通过率和答案状态计算基础奖励分数
    
    Args:
        pass_rate: 采样通过率 (0-16)
        answer_status: "CORRECT", "INCORRECT", or "CLARIFICATION"
    
    Returns:
        float: 基础奖励分数
    """
    # 确保pass_rate在0-16范围内
    pass_rate = max(0, min(16, pass_rate))
    
    if answer_status == "CORRECT":
        # 回答正确曲线：从pass_rate=0时的3分到pass_rate=16时的1分
        # 使用对数曲线，难题（低pass_rate）奖励更高
        if pass_rate <= 1:
            return 3.0
        else:
            # 使用对数衰减
            return 1.0 + 2.0 * (1 - math.log(pass_rate + 1) / math.log(17))
    
    elif answer_status == "INCORRECT":
        # 回答错误曲线：从pass_rate=0时的-1分到pass_rate=16时的-3分
        # 难题错了扣分少，简单题错了扣分多
        if pass_rate >= 15:
            return -3.0
        else:
            # 使用对数增长（负向）
            return -1.0 - 2.0 * (math.log(pass_rate + 1) / math.log(17))
    
    elif answer_status == "CLARIFICATION":
        # 澄清不会曲线：pass_rate=8时为0分，pass_rate=0时为2分，pass_rate=16时为-2分
        if pass_rate <= 8:
            # 从8到0，分数从0增长到2（对数增长）
            if pass_rate == 8:
                return 0.0
            else:
                return 2.0 * (1 - pass_rate / 8.0) ** 0.5
        else:
            # 从8到16，分数从0降到-2（对数下降）
            return -2.0 * ((pass_rate - 8) / 8.0) ** 0.5
    
    return 0.0


def calculate_length_penalty(solution_length: int, pass_rate: float) -> float:
    """
    计算长度惩罚
    
    Args:
        solution_length: 解答文本长度
        pass_rate: 采样通过率 (0-16)
    
    Returns:
        float: 长度惩罚（负数）
    """
    MAX_LENGTH = 8000
    
    # 如果长度在合理范围内，不惩罚
    if solution_length <= 4000:
        return 0.0
    
    # 计算基础惩罚比例
    if solution_length >= MAX_LENGTH:
        base_penalty_ratio = 1.0
    else:
        # 4000-8000字符之间的惩罚，越接近8000惩罚越重
        # 使用指数增长
        ratio = (solution_length - 4000) / 4000
        base_penalty_ratio = (math.exp(ratio * 2) - 1) / (math.exp(2) - 1)
    
    # 根据题目难度调整惩罚力度
    # 简单题目（高pass_rate）惩罚更严重，难题（低pass_rate）惩罚较轻
    difficulty_factor = 0.3 + 0.7 * (pass_rate / 16.0)
    
    # 最终惩罚：最多扣1分（简单题）到0.3分（难题）
    max_penalty = 0.3 + 0.7 * (pass_rate / 16.0)
    
    return -max_penalty * base_penalty_ratio * difficulty_factor


def calculate_reasoning_quality_penalty(quality_result: Dict[str, Any]) -> float:
    """
    计算推理质量惩罚
    
    Args:
        quality_result: 推理质量判断结果
    
    Returns:
        float: 推理质量惩罚（负数或0）
    """
    penalty = 0.0
    
    # 1. 检查未经授权的假设 - 扣2分
    if quality_result.get('has_unauthorized_assumption', False):
        penalty -= 2.0
    
    # 2. 检查推理转折 - 超过1次扣2分
    reversal_count = quality_result.get('reversal_count', 0)
    if reversal_count > 1:
        penalty -= 2.0
    
    return penalty


def compute_score_math_hard(data_source: str, solution_str: str, ground_truth: str, 
                           extra_info: Optional[Dict[str, Any]] = None, **kwargs) -> float:
    """
    计算数学难题的奖励分数
    
    Args:
        data_source: 数据源标识
        solution_str: 模型生成的解答
        ground_truth: 标准答案
        extra_info: 额外信息，应包含 'pass_rate' 和 'ori_question' 字段
        **kwargs: 其他参数（兼容性）
    
    Returns:
        float: 奖励分数
    """
    try:
        # 处理 extra_info 为 None 的情况
        if extra_info is None:
            extra_info = {}
        
        # 获取通过率，默认为8（中等难度）
        pass_rate = extra_info.get('pass_rate', 8.0)
        
        # 确保pass_rate在合理范围内
        pass_rate = float(pass_rate)
        pass_rate = max(0, min(16, pass_rate))
        
        # 检查响应是否为空
        if not solution_str or not solution_str.strip():
            # 空响应视为错误答案
            return calculate_base_reward(pass_rate, "INCORRECT")
        
        # 获取原始问题
        ori_question = extra_info.get('ori_question', '')
        
        # 并发调用两个judge
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交两个任务
            answer_future = executor.submit(check_answer_correctness, solution_str, ground_truth)
            
            if ori_question:
                quality_future = executor.submit(check_reasoning_quality, ori_question, solution_str)
            else:
                quality_future = None
            
            # 获取答案正确性结果
            answer_status = answer_future.result()
            
            # 获取推理质量结果
            quality_result = None
            if quality_future:
                quality_result = quality_future.result()
            else:
                print("Warning: ori_question not provided, skipping reasoning quality check")
        
        # 计算基础奖励
        base_reward = calculate_base_reward(pass_rate, answer_status)
        
        # 计算长度惩罚
        solution_length = len(solution_str)
        length_penalty = calculate_length_penalty(solution_length, pass_rate)
        
        # 计算推理质量惩罚
        reasoning_penalty = 0.0
        if quality_result:
            reasoning_penalty = calculate_reasoning_quality_penalty(quality_result)
        
        # 最终分数 = 基础奖励 + 长度惩罚 + 推理质量惩罚
        final_score = base_reward + length_penalty + reasoning_penalty
        
        # 确保分数在合理范围内 [-8, 3] (因为最多可能扣4分推理质量惩罚)
        final_score = max(-8.0, min(3.0, final_score))
        
        # 调试信息
        if extra_info.get('debug', False):
            print(f"Pass rate: {pass_rate}")
            print(f"Answer status: {answer_status}")
            print(f"Base reward: {base_reward}")
            print(f"Solution length: {solution_length}")
            print(f"Length penalty: {length_penalty}")
            if quality_result:
                print(f"Reasoning quality: {quality_result}")
            print(f"Reasoning penalty: {reasoning_penalty}")
            print(f"Final score: {final_score}")
        
        return final_score
        
    except Exception as e:
        print(f"Error in compute_score_math_hard: {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # 出错时返回0


if __name__ == "__main__":
    # 测试代码
    
    print("="*70)
    print("Testing JSON Extraction:")
    print("="*70)
    
    test_responses = [
        '{"reasoning": "The answer matches", "result": "CORRECT"}',
        '```json\n{"reasoning": "Values differ", "result": "INCORRECT"}\n```',
        'Here is my analysis:\n{"reasoning": "Student needs help", "result": "CLARIFICATION"}',
        'reasoning: "Test", result: INCORRECT',
    ]
    
    for i, response in enumerate(test_responses, 1):
        extracted = extract_json_from_response(response)
        print(f"{i}. Input: {response[:50]}...")
        print(f"   Extracted: {extracted}")
        print()
    
    print("="*70)
    print("Testing Reasoning Quality Judge (Concurrent):")
    print("="*70)
    
    test_question = "What is the area of a rectangle with length 5 and width 3?"
    test_solution_with_issues = """
    Let's assume the rectangle is actually a square for simplicity.
    So the side length would be... wait, that doesn't make sense.
    Actually, let me recalculate. The area is length × width = 5 × 3 = 15.
    
    Answer: 15
    """
    test_ground_truth = "15"
    
    print("Testing concurrent judge calls:")
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交两个任务
        answer_future = executor.submit(check_answer_correctness, test_solution_with_issues, test_ground_truth)
        quality_future = executor.submit(check_reasoning_quality, test_question, test_solution_with_issues)
        
        # 获取结果
        answer_status = answer_future.result()
        quality_result = quality_future.result()
        
        print(f"Answer status: {answer_status}")
        print(f"Quality result: {quality_result}")
    print()
    
    print("="*70)
    print("Testing Full Workflow:")
    print("="*70)
    
    test_solution = """
    To solve this problem, I need to find the value of x.
    
    Let me work through this step by step:
    1. First, I'll set up the equation: 2x + 10 = 94
    2. Subtract 10 from both sides: 2x = 84
    3. Divide by 2: x = 42
    
    Answer: 42
    """
    test_ground_truth = "42"
    test_extra_info = {
        "pass_rate": 8, 
        "ori_question": "Solve for x: 2x + 10 = 94",
        "debug": True
    }
    
    score = compute_score_math_hard(
        data_source="math_hard",
        solution_str=test_solution,
        ground_truth=test_ground_truth,
        extra_info=test_extra_info
    )
    print(f"\nFinal score: {score}")