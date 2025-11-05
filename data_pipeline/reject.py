import json
import re
import time
from typing import Dict, Any, Optional, Tuple
import requests
from pathlib import Path
from datetime import datetime
import logging
import threading
from queue import Queue
from tqdm import tqdm
import asyncio
import aiohttp
from asyncio import Semaphore

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RejectionSampler:
    def __init__(self, api_url: str, model_name: str = "default", max_retries: int = 10, max_concurrent: int = 10):
        """
        初始化拒绝采样器
        
        Args:
            api_url: API地址
            model_name: 模型名称
            max_retries: 最大重试次数
            max_concurrent: 最大并发请求数
        """
        self.api_url = api_url
        self.model_name = model_name
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.semaphore = None  # 将在异步上下文中初始化
        
    async def call_llm_async(self, session: aiohttp.ClientSession, prompt: str, 
                             max_tokens: int = 2048, temperature: float = 0.7) -> Optional[str]:
        """
        异步调用LLM API
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            async with self.semaphore:  # 使用信号量控制并发
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"API调用失败，状态码: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            return None
    
    async def answer_question_async(self, session: aiohttp.ClientSession, question: str) -> Optional[str]:
        """
        异步让LLM回答问题
        """
        prompt = f"""Please answer the following multiple choice question. Provide your answer in the format "The answer is X" where X is the letter of your choice.

Question:
{question}

Please think step by step and provide your final answer."""
        
        response = await self.call_llm_async(session, prompt, temperature=0.7)
        return response
    
    async def check_answer_async(self, session: aiohttp.ClientSession, 
                                 llm_response: str, expected_answer: str) -> Tuple[bool, Dict]:
        """
        异步检查LLM的答案是否正确
        """
        check_prompt = f"""Compare the following two answers and determine if they match. 
Extract the answer choice letter from both responses and check if they are the same.

LLM Response: {llm_response}
Expected Answer: {expected_answer}

Please respond ONLY with a valid JSON object in the following format:
{{
    "llm_answer_letter": "X",
    "expected_answer_letter": "Y",
    "is_correct": true/false,
    "explanation": "brief explanation"
}}

Make sure to extract the letter choice (A, B, C, or D) from both answers."""
        
        for attempt in range(self.max_retries):
            try:
                response = await self.call_llm_async(session, check_prompt, temperature=0.1, max_tokens=200)
                
                if response is None:
                    logger.warning(f"检查答案时API返回None，重试 {attempt + 1}/{self.max_retries}")
                    await asyncio.sleep(2)
                    continue
                
                # 尝试从响应中提取JSON
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    
                    # 验证必要字段
                    if "is_correct" in result:
                        return result.get("is_correct", False), result
                        
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.warning(f"检查答案时出错 (尝试 {attempt + 1}/{self.max_retries}): {e}")
            
            await asyncio.sleep(2)
        
        # 如果所有重试都失败，返回错误
        return False, {"error": "Failed to check answer after all retries"}
    
    async def process_single_sample_async(self, session: aiohttp.ClientSession, 
                                          data: Dict[str, Any], sample_index: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        异步处理单个采样
        
        Returns:
            (成功样本 or None, 失败样本 or None)
        """
        question = data.get("ori_question", "")
        expected_answer = data.get("expected_answer", "")
        
        logger.info(f"处理问题采样 {sample_index}")
        
        # 重试机制
        for retry in range(self.max_retries):
            try:
                # 获取LLM的回答
                llm_response = await self.answer_question_async(session, question)
                
                if llm_response is None:
                    logger.warning(f"LLM响应为空，重试 {retry + 1}/{self.max_retries}")
                    await asyncio.sleep(2)
                    continue
                
                # 检查答案是否正确
                is_correct, check_result = await self.check_answer_async(session, llm_response, expected_answer)
                
                # 准备保存的数据
                sample_data = {
                    **data,
                    "llm_response": llm_response,
                    "sample_index": sample_index,
                    "check_result": check_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                if is_correct:
                    logger.info(f"采样 {sample_index} 成功")
                    return sample_data, None
                else:
                    logger.info(f"采样 {sample_index} 失败")
                    return None, sample_data
                    
            except Exception as e:
                logger.error(f"处理问题时出错 (重试 {retry + 1}/{self.max_retries}): {e}")
                await asyncio.sleep(2)
                
                if retry == self.max_retries - 1:
                    # 最后一次重试仍失败，记录为失败样本
                    return None, {
                        **data,
                        "error": str(e),
                        "sample_index": sample_index,
                        "timestamp": datetime.now().isoformat()
                    }
        
        return None, None


async def run_rejection_sampling_async(
    input_file: str,
    sample_times: int,
    success_output_file: str,
    failed_output_file: str,
    api_url: str,
    model_name: str = "default",
    max_concurrent: int = 10
):
    """
    异步运行拒绝采样主流程
    
    Args:
        input_file: 输入JSONL文件路径
        sample_times: 每个问题的采样次数
        success_output_file: 成功样本输出文件路径
        failed_output_file: 失败样本输出文件路径
        api_url: API地址
        model_name: 模型名称
        max_concurrent: 最大并发请求数（会持续保持这个并发数）
    """
    # 创建采样器
    sampler = RejectionSampler(api_url, model_name, max_concurrent=max_concurrent)
    sampler.semaphore = Semaphore(max_concurrent)  # 初始化信号量
    
    # 读取输入数据
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    questions = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                questions.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效JSON行: {e}")
    
    logger.info(f"加载了 {len(questions)} 个问题")
    logger.info(f"最大并发请求数: {max_concurrent}")
    logger.info(f"总采样数: {len(questions) * sample_times}")
    
    # 创建所有采样任务
    all_tasks = []
    for question_data in questions:
        for i in range(sample_times):
            all_tasks.append((question_data, i + 1))
    
    # 处理所有采样
    all_success_samples = []
    all_failed_samples = []
    
    # 创建 aiohttp session
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建所有协程任务
        tasks = []
        for question_data, sample_idx in all_tasks:
            task = sampler.process_single_sample_async(session, question_data, sample_idx)
            tasks.append(task)
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(tasks), desc="处理采样") as pbar:
            # 使用 as_completed 来跟踪完成的任务
            for coro in asyncio.as_completed(tasks):
                try:
                    success_sample, failed_sample = await coro
                    if success_sample:
                        all_success_samples.append(success_sample)
                    if failed_sample:
                        all_failed_samples.append(failed_sample)
                except Exception as e:
                    logger.error(f"处理采样时出错: {e}")
                pbar.update(1)
    
    # 保存结果
    success_path = Path(success_output_file)
    success_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(success_path, 'w', encoding='utf-8') as f:
        for sample in all_success_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    failed_path = Path(failed_output_file)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(failed_path, 'w', encoding='utf-8') as f:
        for sample in all_failed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 输出统计信息
    logger.info(f"采样完成!")
    logger.info(f"成功样本数: {len(all_success_samples)}")
    logger.info(f"失败样本数: {len(all_failed_samples)}")
    logger.info(f"成功样本保存到: {success_output_file}")
    logger.info(f"失败样本保存到: {failed_output_file}")


def run_rejection_sampling(
    input_file: str,
    sample_times: int,
    success_output_file: str,
    failed_output_file: str,
    api_url: str,
    model_name: str = "default",
    max_concurrent: int = 10
):
    """
    同步包装器，用于运行异步采样
    """
    asyncio.run(run_rejection_sampling_async(
        input_file,
        sample_times,
        success_output_file,
        failed_output_file,
        api_url,
        model_name,
        max_concurrent
    ))


if __name__ == "__main__":
    # 配置参数
    INPUT_JSONL_FILE = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train-medmcqa.jsonl"  # 输入文件路径
    SAMPLE_TIMES = 16  # 每个问题的拒绝采样次数
    SUCCESS_OUTPUT_FILE = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/yitu/medmcqa_all_reject_235b_x16.jsonl"  # 成功样本输出路径
    FAILED_OUTPUT_FILE = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/yitu/medmcqa_all_reject_235b_x16_failed.jsonl"  # 失败样本输出路径
    API_URL = "http://10.80.128.219:9012/v1/chat/completions"
    MODEL_NAME = "default" 
    MAX_CONCURRENT = 10000 
    
    # 运行拒绝采样
    run_rejection_sampling(
        input_file=INPUT_JSONL_FILE,
        sample_times=SAMPLE_TIMES,
        success_output_file=SUCCESS_OUTPUT_FILE,
        failed_output_file=FAILED_OUTPUT_FILE,
        api_url=API_URL,
        model_name=MODEL_NAME,
        max_concurrent=MAX_CONCURRENT
    )