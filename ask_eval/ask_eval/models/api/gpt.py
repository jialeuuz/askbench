import requests
import json
import time
import base64
import asyncio
import aiohttp
import uuid
import logging
from typing import List, Dict, Tuple, Any, Union
from tqdm.asyncio import tqdm
from ask_eval.models.base_api_model import BaseAPIModel

logger = logging.getLogger(__name__)

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
    def __init__(self, url: str, api_type: str = "gpt5", model_name: str = None, sk_token: str = 'none',
                 api_urls: List[str] = None, timeout: float = 300, extra_prompt: str = None,
                 system_prompt: str = None, generate_config: Dict = None):
        normalized_url = self._normalize_url(url)
        normalized_api_urls = [self._normalize_url(item) for item in api_urls] if api_urls else None
        super().__init__(normalized_url, normalized_api_urls, timeout, extra_prompt, system_prompt, generate_config)
        self.api_type = str(api_type or "gpt5")
        resolved_model = model_name or self.api_type or "default"
        self.model_name = str(resolved_model)
        self.sk_token = sk_token or 'none'
        self._model_name_lower = self.model_name.lower()
        self._use_custom_gateway = self._model_name_lower != "default"
        if self._model_name_lower == "default" and self.api_type.lower() != "default":
            self._payload_model = self.api_type
        else:
            self._payload_model = self.model_name
        self.qps = 5
        self.rate_limiter = RateLimiter(self.qps) if self.qps > 0 else None
        self._last_request_time = 0

    @staticmethod
    def _normalize_url(url: str) -> str:
        """确保 GPT 风格的服务始终指向 /chat/completions 或者用户自定义的完整路径"""
        if not url:
            return url
        trimmed = url.rstrip("/")
        lowered = trimmed.lower()
        known_suffixes = (
            "/chat/completions",
            "/responses",
            "/completions",
        )
        if any(lowered.endswith(suffix) for suffix in known_suffixes):
            return trimmed

        normalized = f"{trimmed}/chat/completions"
        logger.info("检测到 GPT API 未包含 chat/completions，已自动补齐: %s -> %s", trimmed, normalized)
        return normalized

    def _encode_image(self, image_path: str) -> str:
        """编码图片为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def _format_request_messages(self, messages: List[Dict], image: str = None,
                                  temperature: float = 0, max_tokens: int = 6000) -> Dict:
        """格式化请求消息体，兼容纯文本与图文混合场景"""
        processed_messages = json.loads(json.dumps(messages))
        image_attached = False

        for msg in processed_messages:
            if msg.get('role') != 'user':
                continue

            content = msg.get('content', '')
            # 仅当提供图片时才将文本转换为多模态格式
            if image and not image_attached:
                multimodal_content = [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image)}"}
                }]

                if isinstance(content, list):
                    multimodal_content.extend(content)
                else:
                    multimodal_content.append({"type": "text", "text": str(content)})

                msg['content'] = multimodal_content
                image_attached = True
            elif not isinstance(content, list):
                msg['content'] = str(content)

        payload = {
            "model": self._payload_model,
            "messages": processed_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        return payload

    def _parse_response_content(self, response_data: Dict) -> str:
        """从响应对象中提取文本内容"""
        if "data" in response_data and isinstance(response_data["data"], dict):
            response_data = response_data["data"]

        choices = response_data.get("choices")
        if not choices:
            raise ValueError(f"响应格式无效: 未找到 'choices' 键。响应内容: {response_data}")

        choice = choices[0]
        content = None

        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
            elif "content" in choice:
                content = choice.get("content")

        if isinstance(content, list):
            # 多模内容仅抽取文本部分
            text_segments = []
            for segment in content:
                if isinstance(segment, dict) and segment.get("type") == "text":
                    text_segments.append(segment.get("text", ""))
            content = "\n".join(filter(None, text_segments))

        if isinstance(content, str):
            return content

        raise ValueError(f"无法从响应中解析内容。Choice: {choice}")

    def _build_headers(self) -> Dict[str, str]:
        """根据当前模型配置生成HTTP请求头"""
        headers = {'Content-Type': 'application/json'}
        if self._use_custom_gateway:
            headers["model"] = self.model_name
            headers["BCS-APIHub-RequestId"] = str(uuid.uuid4())
            if self.sk_token and self.sk_token != 'none':
                headers["X-CHJ-GWToken"] = self.sk_token
        elif self.sk_token and self.sk_token != 'none':
            headers['Authorization'] = f'Bearer {self.sk_token}'
        return headers

    def generate(self, messages: List[Dict[str, str]], 
                max_tokens: int = 6000,
                temperature: float = 0.6,
                image: str = None) -> Tuple[str, str, str]:
        """同步生成响应，支持限速和最多10次重试"""
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

                headers = self._build_headers()
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
                    headers = self._build_headers()
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
