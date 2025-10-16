import requests
import json
import time
import base64
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Any, Union
from tqdm.asyncio import tqdm
from ask_eval.models.base_api_model import BaseAPIModel

# 用于异步QPS限速的辅助类 (无修改)
class RateLimiter:
    """
    一个简单的异步速率限制器。
    """
    def __init__(self, rate: int):
        self.rate = rate
        self.tokens = rate
        self.last_check = time.monotonic()

    async def wait(self):
        while self.tokens < 1:
            self._add_new_tokens()
            await asyncio.sleep(0.05) # 短暂休眠以避免忙等待
        self.tokens -= 1
    
    def _add_new_tokens(self):
        now = time.monotonic()
        time_passed = now - self.last_check
        if time_passed > 0:
            self.tokens = min(self.rate, self.tokens + time_passed * self.rate)
            self.last_check = now

# mind_eval/models/api/gpt4o.py
class GPTAPI(BaseAPIModel):
    """
    GPT API实现，通过api_type支持多种GPT系列模型接口，并包含QPS限速和重试功能。
    
    `api_type` 参数支持以下模型:
    - 'gpt4o': 适配ZNY风格的GPT-4o接口。
    - 'o1-mini', 'o4-mini', 'gpt_41': 适配ZNY风格的其他GPT系列接口。
    - 'gpt5': 适配新的GPT-5接口。  # <--- 新增
    
    `qps` (int):
    - 每秒请求数限制。如果 > 0，将启用速率限制。
    
    API调用失败后将最多重试10次。
    """
    def __init__(self, url: str, api_type: str = "gpt4o", model_name: str = "gpt-4o", api_urls: List[str] = None, 
                 timeout: float = 300, extra_prompt: str = None, system_prompt: str = None, generate_config: Dict = None):
        super().__init__(url, api_urls, timeout, extra_prompt, system_prompt, generate_config)
        self.model_name = model_name
        self.api_type = api_type
        self.qps = 5
        self.rate_limiter = RateLimiter(self.qps) if self.qps > 0 else None
        self._last_request_time = 0

    def _encode_image(self, image_path: str) -> str:
        """编码图片为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def _format_request_messages(self, messages: List[Dict], image: str = None, temperature: float = 0, max_tokens: int = 6000) -> Dict:
        """格式化请求消息体，根据api_type适配不同接口要求"""
        processed_messages = json.loads(json.dumps(messages))

        # 统一处理，确保用户消息的content为列表形式，以支持图文
        for msg in processed_messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if not isinstance(content, list):
                    msg['content'] = [{'type': 'text', 'text': str(content)}]
                
                if image:
                    msg['content'].insert(0, {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image)}"}
                    })
                    image = None

        # 根据api_type调整最终的请求结构
        if self.api_type == 'gpt4o':
            for msg in processed_messages:
                if 'content' in msg:
                    msg['contents'] = msg.pop('content')
            return {
                "messages": processed_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        
        # --- 新增: GPT-5 的请求格式化逻辑 ---
        elif self.api_type == 'gpt5':
            # GPT-5 可能使用 'developer' role 作为 system prompt
            if self.system_prompt:
                # 将 system_prompt 作为第一条消息插入，角色为 'developer'
                processed_messages.insert(0, {"role": "developer", "content": self.system_prompt})
            
            return {
                "model": self.model_name, # 添加 model 字段
                "messages": processed_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        # --- 修改结束 ---

        # 对于 'o1-mini', 'o4-mini', 'gpt_41'，使用标准格式
        else:
            return {
                "messages": processed_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

    def _parse_response_content(self, response_data: Dict) -> str:
        """根据api_type从响应中解析出内容"""
        if self.api_type == 'gpt4o' and "data" in response_data:
            response_data = response_data['data']

        if not response_data.get("choices"):
            raise ValueError(f"响应格式无效: 未找到 'choices' 键。响应内容: {response_data}")

        choice = response_data["choices"][0]

        if self.api_type == 'gpt4o' and "content" in choice:
            return choice["content"]
        # --- 修改: 将 'gpt5' 加入此分支 ---
        # 其他API（o1-mini, gpt5等）的内容在 choice['message']['content']
        elif "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        # --- 修改结束 ---
        else:
            raise ValueError(f"无法从响应中解析内容。Choice: {choice}")

    # 以下的 generate, infer, generate_async, infer_async, infer_batch_async 方法无需任何修改
    # 它们会调用上面已修改的 _format_request_messages 和 _parse_response_content 方法，
    # 从而自动适配新的 'gpt5' 类型。

    def generate(self, messages: List[Dict[str, str]], 
                max_tokens: int = 6000,
                temperature: float = 0.6,
                image: str = None) -> Tuple[str, str, str]:
        """同步生成响应，支持限速和最多10次重试"""
        headers = {'Content-Type': 'application/json'}
        data = self._format_request_messages(messages, image, temperature, max_tokens)
        
        if self.top_k != -1: data['top_k'] = self.top_k
        if self.top_p != -1: data['top_p'] = self.top_p
        
        max_retries = 10
        retries = 0
        while retries < max_retries:
            try:
                if self.qps > 0:
                    now = time.time()
                    time_since_last = now - self._last_request_time
                    wait_time = (1 / self.qps) - time_since_last
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self._last_request_time = time.time()

                response = requests.post(self.url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                response_data = response.json()
                
                content = self._parse_response_content(response_data)
                
                if "</think>" in content:
                    thinking = content.split("</think>")[0].strip() + "</think>"
                    output = content.split("</think>")[1].strip()
                else:
                    thinking = 'none'
                    output = content
                
                return output, thinking, 'not_truncated'
                
            except Exception as e:
                retries += 1
                print(f'API调用异常: {e}。将在1秒后重试 (尝试 {retries}/{max_retries})')
                time.sleep(1)
        
        print(f'API调用达到最大重试次数({max_retries})后失败。')
        return 'wrong data', 'none', 'none'

    def infer(self, message: Union[str, Dict[str, str], List[Dict[str, str]]],
              max_tokens: int = 6000,
              history: List = None,
              sampling_params: Dict = None,
              temperature: float = 0.6) -> Tuple[str, str, str]:
        """同步推理接口"""
        messages = self.format_messages(message, history=history)
        return self.generate(messages=messages, max_tokens=max_tokens, temperature=temperature)


    async def generate_async(self, messages: List[Dict[str, str]],
                           max_tokens: int = 6000,
                           temperature: float = 0.6,
                           image: str = None,
                           url: str = None,
                           timeout: float = None) -> Tuple[str, str, str]:
        """异步生成响应，支持限速和最多10次重试"""
        timeout_value = timeout if timeout is not None else self.timeout
        target_url = url if url else self.url
        
        headers = {'Content-Type': 'application/json'}
        data = self._format_request_messages(messages, image, temperature, max_tokens)
        
        if self.top_k != -1: data['top_k'] = self.top_k
        if self.top_p != -1: data['top_p'] = self.top_p
            
        max_retries = 10
        retries = 0
        backoff_time = 1
        
        while retries < max_retries:
            if self.rate_limiter:
                await self.rate_limiter.wait()

            try:
                timeout_config = aiohttp.ClientTimeout(total=timeout_value)
                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.post(target_url, headers=headers, json=data) as response:
                        response.raise_for_status()
                        response_data = await response.json()
                        
                        content = self._parse_response_content(response_data)
                        
                        if "</think>" in content:
                            thinking = content.split("</think>")[0].strip() + "</think>"
                            output = content.split("</think>")[1].strip()
                        else:
                            thinking = 'none'
                            output = content
                        
                        return output, thinking, 'not_truncated'
                        
            except Exception as e:
                retries += 1
                print(f'异步API调用异常 ({target_url}): {e}。将在{backoff_time}秒后重试 (尝试 {retries}/{max_retries})')
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 30)

        print(f'异步API调用达到最大重试次数({max_retries})后失败 ({target_url})。')
        return 'wrong data', 'none', 'none'

    async def infer_async(self, message: Union[str, Dict[str, str], List[Dict[str, str]]],
                     max_tokens: int = 4096,
                     history: List = None,
                     sampling_params: Dict = None,
                     temperature: float = 0.6,
                     timeout: float = None) -> Tuple[str, str, str]:
        """异步推理接口"""
        timeout_value = timeout if timeout is not None else self.timeout
        
        if isinstance(message, list) and all(isinstance(item, dict) for item in message):
            messages = message
        else:
            messages = self.format_messages(message, history=history)

        return await self.generate_async(
            messages=messages, 
            max_tokens=max_tokens, 
            temperature=temperature,
            timeout=timeout_value
        )

    async def infer_batch_async(self, messages: List[Union[str, Dict[str, str], List[Dict[str, str]]]],
                               max_tokens: int = 4096,
                               temperature: float = 0.6,
                               max_concurrent: int = 15,
                               output_file: str = None,
                               timeout: float = None) -> Tuple[List[str], List[str], List[str]]:
        """异步批量处理"""
        timeout_value = timeout if timeout is not None else self.timeout
        semaphore = asyncio.Semaphore(max_concurrent)
        formatted_messages = [self.format_messages(message) for message in messages]
        
        async def process_single_message(message_list: List[Dict[str, str]], idx: int) -> str:
            url_to_use = self.api_urls[idx % len(self.api_urls)]
            
            async with semaphore:
                return await self.generate_async(
                    messages=message_list,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    url=url_to_use,
                    timeout=timeout_value
                )
        
        tasks = [process_single_message(msg, i) for i, msg in enumerate(formatted_messages)]
        
        results = await tqdm.gather(*tasks, desc=f"Processing messages with {len(self.api_urls)} URLs", total=len(formatted_messages))
        
        responses, thinking_processes, truncated_flags = [], [], []
        for output, thinking, truncated in results:
            responses.append(output)
            thinking_processes.append(thinking)
            truncated_flags.append(truncated)
            
        return responses, thinking_processes, truncated_flags