# data_pipeline/utils/base_api_model.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Union
import asyncio
from tqdm.asyncio import tqdm
from utils.check_url_health import check_urls_health


class BaseAPIModel(ABC):
    """Base class for OpenAI-compatible API models."""
    def __init__(self, url: str, api_urls: List[str] = None, timeout: float = 600, extra_prompt: str = None, system_prompt: str = None, generate_config: Dict = None):
        self.url = url
        self.api_urls = api_urls or [url]  # fall back to a single URL
        self.timeout = timeout  # default request timeout
        self.extra_prompt = extra_prompt  # optional suffix appended to user content
        self.system_prompt = system_prompt  # optional system message
        self.generate_config = generate_config  # generation params
        if generate_config:
            self.top_k = generate_config.get("top_k", -1)
            self.top_p = generate_config.get("top_p", -1)
        else:
            self.top_k = -1
            self.top_p = -1

    def check_health(self, max_wait_minutes: int = 15) -> bool:
        """
        Check whether any configured API URL is reachable/healthy.
        
        Args:
            max_wait_minutes: maximum time to wait (minutes)
            
        Returns:
            bool: True if at least one URL is healthy
        """
        # Grab auth/model params if present on the concrete implementation.
        sk_token = getattr(self, 'sk_token', None)
        api_type = getattr(self, 'api_type', None)
        
        # Probe all URLs.
        success, healthy_urls = check_urls_health(
            self.api_urls,
            sk_token=sk_token,
            api_type=api_type,
            max_wait_minutes=max_wait_minutes
        )
        
        # If any URL is healthy, keep only healthy URLs.
        if success and healthy_urls:
            self.api_urls = healthy_urls
            print("Detected healthy URLs; updating api_urls:", healthy_urls)
            # If the primary URL is unhealthy, switch to the first healthy URL.
            if self.url not in healthy_urls:
                self.url = healthy_urls[0]
                print("Updated primary URL:", self.url)
        return success

    @abstractmethod
    async def generate_async(self, messages: List[Dict[str, str]],
                           max_retries: int = 4,
                           max_tokens: int = 6000,
                           temperature: float = 0.6,
                           url: str = None,
                           timeout: float = None) -> str:
        """Asynchronously generate a response."""
        pass

    @abstractmethod
    async def infer_batch_async(self, messages: List[Union[str, Dict[str, str], List[Dict[str, str]]]],
                               max_tokens: int = 4096,
                               temperature: float = 0.6,
                               max_concurrent: int = 15,
                               output_file: str = None,
                               timeout: float = None) -> List[str]:
        """Asynchronously process a batch."""
        pass

    def format_messages(self, message: Union[str, Dict[str, str], List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Convert supported message inputs to a standard list-of-messages format.
        
        Args:
            message: a string, a single {"role","content"} dict, or a list of such dicts
            
        Returns:
            List[Dict[str, str]]: standardized message list
        """
        # Whether to append extra/system prompts.
        has_extra = hasattr(self, 'extra_prompt') and self.extra_prompt
        has_system = hasattr(self, 'system_prompt') and self.system_prompt
        
        result = []
        
        # Add system prompt (if provided).
        if has_system:
            result.append({'role': 'system', 'content': self.system_prompt})
        
        if isinstance(message, dict) and 'role' in message and 'content' in message:
            # Single message dict
            if has_extra and message['role'] == 'user':
                message = message.copy()  # avoid mutating the input object
                # If message['content'] is already a list (multimodal), do not append extra_prompt.
                if isinstance(message['content'], list):
                    pass
                else:
                    message['content'] = f"{message['content']}\n{self.extra_prompt}"
            result.append(message)
        
        elif isinstance(message, list):
            # List of message dicts
            if has_extra:
                for item in message:
                    item_copy = item.copy()  # avoid mutating the input object
                    if item_copy.get('role') == 'user':
                        item_copy['content'] = f"{item_copy['content']}\n{self.extra_prompt}"
                    result.append(item_copy)
            else:
                result.extend(message)
        
        else:
            # String message
            content = str(message)
            if has_extra:
                content = f"{content}\n{self.extra_prompt}"
            result.append({'role': 'user', 'content': content})
        return result
