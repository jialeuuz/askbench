# ask_eval/models/api/custom.py
import requests
import json
import asyncio
import aiohttp
import time
from typing import List, Dict, Tuple, Any, Union
from tqdm.asyncio import tqdm
from ask_eval.models.base_api_model import BaseAPIModel

class CustomAPI(BaseAPIModel):
    def __init__(self, url: str, sk_token: str = 'none', api_type: str = 'custom-reasoner', 
                 api_urls: List[str] = None, timeout: float = 600, extra_prompt: str = None, system_prompt: str = None, enable_thinking: bool = True, generate_config: Dict = None):
        super().__init__(url, api_urls, timeout, extra_prompt, system_prompt, generate_config)
        self.sk_token = sk_token
        self.api_type = api_type
        self.enable_thinking = enable_thinking

    def _parse_content(self, content: str, response_data: Dict) -> Tuple[str, str, str]:
        """Helper function to parse content and extract thinking, output, and truncated status."""
        # Extract model output and thinking process
        if "</think>" in content:
            parts = content.split("</think>", 1)
            thinking = parts[0].strip() + "</think>"
            output = parts[1].strip()
        else:
            thinking = 'none'
            output = content
        
        # Check truncation status
        truncated = 'none'
        if 'finish_reason' in response_data.get('choices', [{}])[0]:
            truncated = 'not_truncated' if response_data['choices'][0]['finish_reason'] == 'stop' else 'truncated'
        
        return output, thinking, truncated

    def generate(self, messages: List[Dict[str, str]], 
                max_retries: int = 6,
                max_tokens: int = 6000,
                temperature: float = 0.6) -> Tuple[str, str, str]:
        """同步生成响应"""
        headers = {'Content-Type': 'application/json'}
        data_entry = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
        }

        if self.sk_token != 'none':
            headers['Authorization'] = f'Bearer {self.sk_token}'
        if self.api_type != 'none':
            data_entry['model'] = self.api_type
        if self.top_k != -1:
            data_entry['top_k'] = self.top_k
        if self.top_p != -1:
            data_entry['top_p'] = self.top_p
            
        retries = 0
        while retries < max_retries:
            response = None  # Initialize response to None
            try:
                response = requests.request("POST", self.url, headers=headers, json=data_entry, timeout=self.timeout)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    if 'choices' not in response_data or not response_data['choices']:
                        print(f"API响应成功，但内容不符合预期: {response.text}")
                        raise ValueError("API response is missing 'choices' or 'choices' is empty.")

                    content = str(response_data['choices'][0]['message']['content'])
                    return self._parse_content(content, response_data)
                
                else:
                    # Handle non-200 status codes
                    print(f"API请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
                    # Raise an exception to trigger the retry logic
                    response.raise_for_status()

            except requests.exceptions.RequestException as e:
                print(f'API网络或HTTP错误：{e}')
                # The response object might be available here with error details
                if response is not None:
                    print(f"原始API响应: {response.text}")
                time.sleep(1)
                retries += 1
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Handle cases where response is 200 but content is not valid JSON or lacks expected keys
                print(f'API响应解析异常：{e}')
                if response is not None:
                    print(f"原始API响应: {response.text}")
                time.sleep(1)
                retries += 1
            except Exception as e:
                print(f'未预料的API调用异常：{e}')
                if response is not None and hasattr(response, 'text'):
                    print(f"原始API响应: {response.text}")
                time.sleep(1)
                retries += 1

        print("超过最大尝试次数！")
        return 'Error', 'none', 'none'

    def infer(self, message: Union[str, Dict[str, str], List[Dict[str, str]]],
              max_tokens: int = 6000,
              history: List = None,
              sampling_params: Dict = None,
              temperature: float = 0.6) -> Tuple[str, str, str]:
        """同步推理接口"""
        messages = self.format_messages(message)
        return self.generate(
            messages=messages, 
            max_tokens=max_tokens, 
            temperature=temperature
        )

    async def generate_async(self, messages: List[Dict[str, str]],
                           max_retries: int = 4,
                           max_tokens: int = 6000,
                           temperature: float = 0.6,
                           url: str = None,
                           timeout: float = None) -> Tuple[str, str, str]:
        """异步生成响应，支持指定URL"""
        timeout_value = timeout if timeout is not None else self.timeout
        target_url = url if url else self.url
        
        headers = {'Content-Type': 'application/json'}
        data_entry = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
        }

        if self.sk_token != 'none':
            headers['Authorization'] = f'Bearer {self.sk_token}'
        if self.api_type != 'none':
            data_entry['model'] = self.api_type
        if self.top_k != -1:
            data_entry['top_k'] = self.top_k
        if self.top_p != -1:
            data_entry['top_p'] = self.top_p
            
        retries = 0
        backoff_time = 1  # 初始退避时间（秒）
        
        while retries < max_retries:
            response_text = ""
            try:
                timeout_config = aiohttp.ClientTimeout(
                    total=timeout_value, connect=60, sock_connect=60, sock_read=timeout_value
                )
                
                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.post(target_url, headers=headers, json=data_entry) as response:
                        response_text = await response.text()
                        
                        if response.status == 200:
                            response_data = json.loads(response_text)
                            
                            if 'choices' not in response_data or not response_data['choices']:
                                print(f"API响应成功 ({target_url})，但内容不符合预期: {response_text}")
                                raise ValueError("API response is missing 'choices' or 'choices' is empty.")

                            content = str(response_data['choices'][0]['message']['content'])
                            return self._parse_content(content, response_data)
                        else:
                            print(f"API请求失败 ({target_url})，状态码: {response.status}, 响应内容: {response_text}")
                            # Raise an exception to trigger retry logic
                            response.raise_for_status()

            except asyncio.TimeoutError:
                print(f'API请求超时 ({target_url})：超过{timeout_value}秒未响应')
                retries += 1
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 30)
                
            except aiohttp.ClientError as e:
                print(f'API网络或HTTP错误 ({target_url})：{e}')
                if response_text:
                    print(f"原始API响应: {response_text}")
                retries += 1
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 30)
            
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f'API响应解析异常 ({target_url})：{e}')
                if response_text:
                    print(f"原始API响应: {response_text}")
                retries += 1
                await asyncio.sleep(1)

            except Exception as e:
                print(f'未预料的API调用异常 ({target_url})：{e}')
                if response_text:
                    print(f"原始API响应: {response_text}")
                retries += 1
                await asyncio.sleep(1)

        print(f"超过最大尝试次数！URL: {target_url}")
        return 'Error', 'none', 'none'

    async def infer_async(self, message: Union[str, Dict[str, str], List[Dict[str, str]]],
                    max_tokens: int = 4096,
                    history: List = None, # 保留 history 参数
                    sampling_params: Dict = None,
                    temperature: float = 0.6,
                    timeout: float = None) -> Tuple[str, str, str]:
        """
        异步推理接口。
        能处理简单的字符串/字典输入，也能直接接受一个完整的对话历史列表。
        """
        # 核心修正：检查 message 是否已经是我们需要的格式
        if isinstance(message, list) and all(isinstance(item, dict) for item in message):
            # 如果 message 已经是 List[Dict] 格式 (即对话历史)，直接使用它
            messages = message
        else:
            # 否则，使用 format_messages 来处理简单的输入，保持向后兼容
            messages = self.format_messages(message, history=history)

        return await self.generate_async(
            messages=messages, 
            max_tokens=max_tokens, 
            temperature=temperature,
            timeout=timeout
        )

    async def infer_batch_async(self, messages: List[Union[str, Dict[str, str], List[Dict[str, str]]]],
                              max_tokens: int = 4096,
                              temperature: float = 0.6,
                              max_concurrent: int = 15,
                              output_file: str = None,
                              timeout: float = None) -> Tuple[List[str], List[str], List[str]]:
        """异步批量处理，支持多URL"""
        timeout_value = timeout if timeout is not None else self.timeout
        
        semaphore = asyncio.Semaphore(max_concurrent)
        formatted_messages = [self.format_messages(message) for message in messages]
        
        async def process_single_message(message_list: List[Dict[str, str]], idx: int) -> Tuple[str, str, str]:
            # 根据消息索引循环选择URL
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