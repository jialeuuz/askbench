# verl/utils/reward_score/multiturn_qa.py
import os
import random
import time
from typing import List, Tuple
import requests
import json
import re

# 从环境变量获取API URLs，如果没有则使用默认值
API_URLS = [
    "http://10.72.0.14:8001/v1/chat/completions"
]

import time


def call_llm_api(prompt: str, max_retries: int = 10) -> str:
    """调用LLM API进行判断"""
    for attempt in range(max_retries):
        try:
            api_urls = API_URLS
            
            if not api_urls:
                print("No API URLs available")
                time.sleep(1)
                continue
            
            # 随机选择一个API
            api_url = random.choice(api_urls)
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "grader",
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator for multi-turn conversational QA systems. You assess whether responses appropriately gather information or provide answers based on context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return content
            else:
                print(f"API call failed with status {response.status_code}, retrying with a different API...")
                
        except requests.exceptions.Timeout:
            time.sleep(0.5)
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
        except Exception as e:
            time.sleep(0.5)
    
    print(f"All {max_retries} attempts failed, returning default score")
    return "0"

def format_conversation_history(context: str) -> str:
    """格式化对话历史，使其更易读"""
    if not context:
        return "No previous conversation."
    return context

def evaluate_non_final_turn(ori_question: str, degraded_info: str, current_question: str, 
                            context: str, response: str) -> float:
    """评估非最终轮次的响应质量
    
    Args:
        ori_question: 原始完整问题
        degraded_info: 被降级/缺失的信息
        current_question: 当前轮次的问题
        context: 之前的对话历史
        response: 模型的响应
    """
    
    conversation_history = format_conversation_history(context)
    
    prompt = f"""You are evaluating an AI assistant's response in a multi-turn conversation where information is missing.

Original Complete Question (with all information):
{ori_question}

Missing/Degraded Information (what the user didn't provide):
{degraded_info}

Previous Conversation History:
{conversation_history}

Current User Question:
{current_question}

Assistant's Response:
{response}

Scoring Guidelines:
1. Score = -2.0 (Critical Failure): The assistant provides a final answer without asking for the missing information
   - This is the worst case: the model should learn to ask clarifying questions when information is missing
   - Example: User asks incomplete math problem, assistant gives answer without asking for missing parameters
   
2. Score = -0.8 (Poor Clarification): The assistant asks questions BUT:
   - Misses ALL the key missing information points
   - Asks about irrelevant or already-provided information
   - Shows it didn't understand what information is actually missing
   - Example: Asks about tangential details while ignoring critical missing data
   
3. Score = 0.8 (Good Clarification): The assistant asks about at least ONE key missing point:
   - Identifies at least some of the missing information correctly
   - Shows understanding that more information is needed
   - Questions are relevant to the degraded information
   - Example: Asks for one of two missing parameters in a math problem
   
4. Score = 1.0 (Excellent Clarification): The assistant comprehensively asks about ALL key missing points:
   - Identifies all or nearly all missing information
   - Asks targeted, relevant questions
   - Shows complete understanding of what's needed to answer properly
   - Example: Systematically asks for all missing variables/conditions needed

Important: Consider the conversation history. If previous turns already clarified some information, the assistant should focus on remaining gaps.

Output only a single number (-2.0, -0.8, 0.8, or 1.0) based on the guidelines above.
"""
    
    result = call_llm_api(prompt)
    try:
        # 提取数字
        score_match = re.search(r'-?\d+\.?\d*', result)
        if score_match:
            score = float(score_match.group())
            # 确保分数在允许的值中
            if score <= -1.5:
                return -2.0
            elif score <= -0.4:
                return -0.8
            elif score >= 0.9:
                return 1.0
            else:
                return 0.8
    except:
        pass
    
    return -0.8  # 默认值

def evaluate_final_turn(expected_answer: str, current_question: str, 
                       context: str, response: str) -> float:
    """评估最终轮次的答案质量
    
    Args:
        expected_answer: 期望的正确答案
        current_question: 当前轮次的问题
        context: 之前的对话历史
        response: 模型的响应
    """
    
    conversation_history = format_conversation_history(context)
    
    prompt = f"""You are evaluating an AI assistant's final answer in a multi-turn conversation.

Previous Conversation History:
{conversation_history}

Current User Question (asking for final answer):
{current_question}

Expected Correct Answer:
{expected_answer}

Assistant's Response:
{response}

Scoring Guidelines:
1. Score = -2.0 (Critical Failure): The assistant is still asking clarifying questions
   - At this point, all necessary information has been provided
   - The model should have learned when to stop asking and provide an answer
   - This is the worst case: not recognizing when enough information is available
   - Example: User has provided all needed data, but assistant asks for more details
   
2. Score = -1.0 (Wrong Answer): The assistant provides an answer BUT:
   - The answer is factually incorrect
   - The answer contradicts the expected answer
   - The answer is off-topic or doesn't address the question
   - The answer misinterprets the information from the conversation
   - Example: Math calculation error, wrong logical conclusion, misunderstanding of the problem
   
3. Score = 1.0 (Correct Answer): The assistant provides a correct answer:
   - The core conclusion matches the expected answer
   - The answer is based on the information gathered in the conversation
   - Minor wording differences are acceptable if the meaning is correct
   - The answer appropriately addresses the user's question
   - Example: Correct calculation, valid reasoning, accurate conclusion

Important: 
- Consider the full conversation context when evaluating correctness
- Focus on factual accuracy and whether the answer is essentially correct
- The answer should be logically consistent with the information gathered in previous turns
- There is no partial credit: it's either correct (1.0) or wrong (-1.0/-2.0)

Output only a single number (-2.0, -1.0, or 1.0) based on the guidelines above.
"""
    
    result = call_llm_api(prompt)
    try:
        # 提取数字
        score_match = re.search(r'-?\d+\.?\d*', result)
        if score_match:
            score = float(score_match.group())
            # 确保分数在允许的值中
            if score <= -1.5:
                return -2.0
            elif score <= 0:
                return -1.0
            else:
                return 1.0
    except:
        pass
    
    return -1.0  # 默认值

def compute_score_medical_qa(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    计算多轮对话QA的奖励分数（通用版本，适用于数学、医疗、编程等各类领域）
    
    Args:
        data_source: 数据源标识
        solution_str: 模型生成的响应
        ground_truth: 标准答案
        extra_info: 额外信息，包含是否最终轮次、原始问题、上下文等
        **kwargs: 其他参数（兼容性）
    
    Returns:
        float: 奖励分数
        
    Score Mechanism:
        Non-final turns (information gathering):
            -2.0: Provides final answer when information is missing
            -0.8: Asks questions but misses all key missing points
             0.8: Asks about at least one key missing point
             1.0: Comprehensively asks about all key missing points
             
        Final turn (answering):
            -2.0: Still asking questions when should answer
            -1.0: Provides wrong answer
             1.0: Provides correct answer
    """
    try:
        # 处理 extra_info 为 None 的情况
        if extra_info is None:
            extra_info = {}
        
        # 解析extra_info
        is_final_turn = extra_info.get('is_final_turn', True)
        ori_question = extra_info.get('ori_question', '')
        degraded_info = extra_info.get('degraded_info', '')
        expected_answer = extra_info.get('expected_answer', '')
        current_question = extra_info.get('question', '')  # 当前轮次的问题
        context = extra_info.get('context', '')  # 对话历史
        
        # 检查响应是否为空
        if not solution_str or not solution_str.strip():
            return -2.0
        
        if is_final_turn:
            # 最终轮次 - 必须有标准答案
            if not expected_answer:
                print("Warning: Final turn without expected answer, using ground_truth")
                expected_answer = ground_truth
            
            reward_score = evaluate_final_turn(
                expected_answer=expected_answer,
                current_question=current_question,
                context=context,
                response=solution_str
            )
            
        else:
            # 非最终轮次 - 必须有退化信息
            if not degraded_info:
                print("Warning: Non-final turn without degraded info")
                return -2.0
            
            reward_score = evaluate_non_final_turn(
                ori_question=ori_question,
                degraded_info=degraded_info,
                current_question=current_question,
                context=context,
                response=solution_str
            )
        
        return reward_score
        
    except Exception as e:
        print(f"Error in compute_score_medical_qa: {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # 出错时返回0


# 向后兼容的别名
compute_score_multiturn_qa = compute_score_medical_qa


if __name__ == "__main__":
    # 测试代码
    print("Testing API URLs loading...")
    res = load_api_urls_from_env()
    print("Loaded API URLs:", res)
    
    # 测试非最终轮次评估 - 数学示例
    print("\n" + "="*50)
    print("Testing Non-Final Turn Evaluation (Math Example)...")
    print("="*50)
    test_score = evaluate_non_final_turn(
        ori_question="Solve for x: 2x + 3 = 11, where x must be a positive integer.",
        degraded_info="The constraint that x must be a positive integer is missing.",
        current_question="Solve for x: 2x + 3 = 11",
        context="",
        response="I need to know: are there any constraints on what x can be? For example, should x be an integer, positive, or within a certain range?"
    )
    print(f"Test score (should be 1.0): {test_score}")
    
    # 测试非最终轮次评估 - 医疗示例
    print("\n" + "="*50)
    print("Testing Non-Final Turn Evaluation (Medical Example)...")
    print("="*50)
    test_score = evaluate_non_final_turn(
        ori_question="I have a severe headache that started 2 hours ago after I hit my head.",
        degraded_info="The timing (2 hours ago) and cause (hit head) are missing.",
        current_question="I have a headache.",
        context="",
        response="Can you tell me when the headache started and if there was any injury or trauma?"
    )
    print(f"Test score (should be 1.0): {test_score}")
    
    # 测试最终轮次评估 - 数学示例
    print("\n" + "="*50)
    print("Testing Final Turn Evaluation (Math Example)...")
    print("="*50)
    test_score = evaluate_final_turn(
        expected_answer="x = 4",
        current_question="Now give me the final answer.",
        context="User: Solve for x: 2x + 3 = 11\nAssistant: Are there any constraints?\nUser: x must be a positive integer.",
        response="Solving the equation: 2x + 3 = 11, we get 2x = 8, so x = 4."
    )
    print(f"Test score (should be 1.0): {test_score}")
    
    # 测试最终轮次评估 - 医疗示例
    print("\n" + "="*50)
    print("Testing Final Turn Evaluation (Medical Example)...")
    print("="*50)
    test_score = evaluate_final_turn(
        expected_answer="Based on your symptoms, you should see a doctor immediately as head injuries can be serious.",
        current_question="What should I do?",
        context="User: I have a headache.\nAssistant: When did it start?\nUser: 2 hours ago after I hit my head.",
        response="You should see a doctor immediately as head injuries require medical attention."
    )
    print(f"Test score (should be 1.0): {test_score}")