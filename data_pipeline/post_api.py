# data_pipeline/post_api.py
import requests
import json
import asyncio
import aiohttp
import time
from typing import List, Dict, Tuple, Any, Union
from tqdm.asyncio import tqdm
from utils.base_api_model import BaseAPIModel

class CustomAPI(BaseAPIModel):
    def __init__(self, url: str, sk_token: str = 'none', api_type: str = 'custom-reasoner', 
                 api_urls: List[str] = None, timeout: float = 600, extra_prompt: str = None, system_prompt: str = None, enable_thinking: bool = True, generate_config: Dict = None):
        super().__init__(url, api_urls, timeout, extra_prompt, system_prompt, generate_config)
        self.sk_token = sk_token
        self.api_type = api_type
        self.enable_thinking = enable_thinking
        self.timeout = 3600

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

    async def generate_async(self, messages: List[Dict[str, str]],
                           max_retries: int = 20,
                           max_tokens: int = 28000,
                           temperature: float = 0.6,
                           url: str = None,
                           timeout: float = 3600) -> Tuple[str, str, str]:
        """Asynchronously generate a response (supports overriding the target URL)."""
        timeout_value = timeout if timeout is not None else self.timeout
        target_url = url if url else self.url
        
        headers = {'Content-Type': 'application/json'}
        data_entry = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # low, medium, high

        if self.sk_token != 'none':
            headers['Authorization'] = f'Bearer {self.sk_token}'
        if self.api_type != 'none':
            data_entry['model'] = self.api_type
        if self.top_k != -1:
            data_entry['top_k'] = self.top_k
        if self.top_p != -1:
            data_entry['top_p'] = self.top_p
            
        retries = 0
        backoff_time = 1  # initial backoff (seconds)
        
        while retries < max_retries:
            response_text = ""
            try:
                # timeout_config = aiohttp.ClientTimeout(
                #     total=timeout_value, connect=60, sock_connect=60, sock_read=timeout_value
                # )
                timeout_config = aiohttp.ClientTimeout(total=timeout_value)
                
                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.post(target_url, headers=headers, json=data_entry) as response:
                        response_text = await response.text()
                        
                        if response.status == 200:
                            response_data = json.loads(response_text)
                            
                            if 'choices' not in response_data or not response_data['choices']:
                                print(f"API call succeeded ({target_url}) but response is missing expected fields: {response_text}")
                                raise ValueError("API response is missing 'choices' or 'choices' is empty.")

                            content = str(response_data['choices'][0]['message']['content'])
                            return self._parse_content(content, response_data)
                        else:
                            # print(f"API request failed ({target_url}). status={response.status}, body={response_text}")
                            # Raise an exception to trigger retry logic
                            response.raise_for_status()

            except asyncio.TimeoutError:
                print(f"API request timed out ({target_url}): no response within {timeout_value} seconds")
                retries += 1
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 30)
                
            except aiohttp.ClientError as e:
                # print(f"Network/HTTP error ({target_url}): {e}")
                if response_text:
                    # print(f"Raw API response: {response_text}")
                    pass
                retries += 1
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 30)
            
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Failed to parse API response ({target_url}): {e}")
                if response_text:
                    # print(f"Raw API response: {response_text}")
                    pass
                retries += 1
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Unexpected API call error ({target_url}): {e}")
                if response_text:
                    print(f"Raw API response: {response_text}")
                retries += 1
                await asyncio.sleep(1)

        print(f"Exceeded maximum retry attempts. URL: {target_url}")
        return 'Error', 'none', 'none'

    async def infer_batch_async(self, messages: List[Union[str, Dict[str, str], List[Dict[str, str]]]],
                              max_tokens: int = 4096,
                              temperature: float = 0.6,
                              max_concurrent: int = 15,
                              output_file: str = None,
                              timeout: float = None) -> Tuple[List[str], List[str], List[str]]:
        """Asynchronously process a batch of inputs (supports multiple URLs)."""
        timeout_value = timeout if timeout is not None else self.timeout
        
        semaphore = asyncio.Semaphore(max_concurrent)
        formatted_messages = [self.format_messages(message) for message in messages]
        
        async def process_single_message(message_list: List[Dict[str, str]], idx: int) -> Tuple[str, str, str]:
            # Pick a URL in round-robin fashion based on the message index.
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
