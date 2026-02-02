# mind_eval/models/base_api_model.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Union
import asyncio
from tqdm.asyncio import tqdm
from ask_eval.utils.url_health import check_urls_health


class BaseAPIModel(ABC):
    """Base class for API-backed models."""
    def __init__(self, url: str, api_urls: List[str] = None, timeout: float = 600, extra_prompt: str = None, system_prompt: str = None, generate_config: Dict = None):
        self.url = url
        self.api_urls = api_urls or [url]  # If api_urls is not provided, fall back to the single url
        self.timeout = timeout  # Default request timeout
        self.extra_prompt = extra_prompt  # Optional extra prompt to append to user messages
        self.system_prompt = system_prompt  # Optional system prompt
        self.generate_config = generate_config  # Optional generation config
        if generate_config:
            self.top_k = generate_config.get("top_k", -1)
            self.top_p = generate_config.get("top_p", -1)
        else:
            self.top_k = -1
            self.top_p = -1

    def check_health(self, max_wait_minutes: int = 15) -> bool:
        """
        Check whether at least one API URL is healthy.
        
        Args:
            max_wait_minutes: maximum wait time (minutes)
            
        Returns:
            bool: whether any URL is healthy
        """
        # Get auth/config fields if present.
        sk_token = getattr(self, 'sk_token', None)
        api_type = getattr(self, 'api_type', None)
        model_name = getattr(self, 'model_name', None)

        # Check all URLs
        success, healthy_urls = check_urls_health(
            self.api_urls,
            sk_token=sk_token,
            api_type=api_type,
            model_name=model_name,
            max_wait_minutes=max_wait_minutes
        )
        
        # If there are healthy URLs, keep only those and update the primary URL if needed.
        if success and healthy_urls:
            self.api_urls = healthy_urls
            print("Detected healthy URLs; updated api_urls:", healthy_urls)
            # If the primary URL is unhealthy, switch to the first healthy one.
            if self.url not in healthy_urls:
                self.url = healthy_urls[0]
                print("Updated primary URL:", self.url)
        return success
        
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], 
                max_retries: int = 6,
                max_tokens: int = 6000,
                temperature: float = 0.6) -> Tuple[str, Dict]:
        """Generate a response synchronously."""
        pass

    @abstractmethod
    def infer(self, message: Union[str, Dict[str, str], List[Dict[str, str]]],
              max_tokens: int = 6000,
              history: List = None,
              sampling_params: Dict = None,
              temperature: float = 0.6) -> Tuple[str, Dict]:
        """Synchronous inference interface."""
        pass

    @abstractmethod
    async def generate_async(self, messages: List[Dict[str, str]],
                           max_retries: int = 4,
                           max_tokens: int = 6000,
                           temperature: float = 0.6,
                           url: str = None,
                           timeout: float = None) -> str:
        """Generate a response asynchronously."""
        pass

    @abstractmethod
    async def infer_async(self, message: Union[str, Dict[str, str], List[Dict[str, str]]],
                         max_tokens: int = 4096,
                         history: List = None,
                         sampling_params: Dict = None,
                         temperature: float = 0.6,
                         timeout: float = None) -> Tuple[str, Dict]:
        """Asynchronous inference interface."""
        pass

    @abstractmethod
    async def infer_batch_async(self, messages: List[Union[str, Dict[str, str], List[Dict[str, str]]]],
                               max_tokens: int = 4096,
                               temperature: float = 0.6,
                               max_concurrent: int = 15,
                               output_file: str = None,
                               timeout: float = None) -> List[str]:
        """Asynchronous batched inference."""
        pass

    def format_messages(self, message: Union[str, Dict[str, str], List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Convert multiple message input formats into a standard list-of-messages format.
        
        Args:
            message: can be a string, a single message dict, or a list of message dicts
            
        Returns:
            A standardized message list
        """
        # Determine whether to append extra_prompt and/or system_prompt.
        has_extra = hasattr(self, 'extra_prompt') and self.extra_prompt
        has_system = hasattr(self, 'system_prompt') and self.system_prompt
        
        result = []
        
        # Add system prompt (if any)
        if has_system:
            result.append({'role': 'system', 'content': self.system_prompt})
        
        if isinstance(message, dict) and 'role' in message and 'content' in message:
            # Single message dict
            if has_extra and message['role'] == 'user':
                message = message.copy()  # Copy to avoid mutating the original object
                # If the passed-in massage['content'] is a list, do not append extra_prompt.
                if isinstance(message['content'], list):
                    pass
                else:
                    message['content'] = f"{message['content']}\n{self.extra_prompt}"
            result.append(message)
        
        elif isinstance(message, list):
            # List of message dicts
            if has_extra:
                for item in message:
                    item_copy = item.copy()  # Copy
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
