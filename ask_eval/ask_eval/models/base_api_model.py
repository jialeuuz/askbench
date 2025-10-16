# mind_eval/models/base_api_model.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Union
import asyncio
from tqdm.asyncio import tqdm
from ask_eval.utils.url_health import check_urls_health


class BaseAPIModel(ABC):
    """API模型基类"""
    def __init__(self, url: str, api_urls: List[str] = None, timeout: float = 600, extra_prompt: str = None, system_prompt: str = None, generate_config: Dict = None):
        self.url = url
        self.api_urls = api_urls or [url]  # 如果未提供api_urls，则使用单个url
        self.timeout = timeout  # 添加默认超时时间
        self.extra_prompt = extra_prompt  # 添加额外提示
        self.system_prompt = system_prompt  # 添加系统提示
        self.generate_config = generate_config  # 添加生成配置
        if generate_config:
            self.top_k = generate_config.get("top_k", -1)
            self.top_p = generate_config.get("top_p", -1)
        else:
            self.top_k = -1
            self.top_p = -1

    def check_health(self, max_wait_minutes: int = 15) -> bool:
        """
        检查API URL是否健康
        
        Args:
            max_wait_minutes: 最长等待时间（分钟）
            
        Returns:
            bool: 是否有健康的URL
        """
        # 获取sk_token和api_type
        sk_token = getattr(self, 'sk_token', None)
        api_type = getattr(self, 'api_type', None)
        
        # 检查所有URLs
        success, healthy_urls = check_urls_health(
            self.api_urls,
            sk_token=sk_token,
            api_type=api_type,
            max_wait_minutes=max_wait_minutes
        )
        
        # 如果有健康的URL，更新api_urls只保留健康的
        if success and healthy_urls:
            self.api_urls = healthy_urls
            print("检测所有健康的URL，更新api_urls: ", healthy_urls)
            # 如果主URL不健康，将第一个健康URL设为主URL
            if self.url not in healthy_urls:
                self.url = healthy_urls[0]
                print("更新主URL: ", self.url)
        return success
        
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], 
                max_retries: int = 6,
                max_tokens: int = 6000,
                temperature: float = 0.6) -> Tuple[str, Dict]:
        """同步生成响应"""
        pass

    @abstractmethod
    def infer(self, message: Union[str, Dict[str, str], List[Dict[str, str]]],
              max_tokens: int = 6000,
              history: List = None,
              sampling_params: Dict = None,
              temperature: float = 0.6) -> Tuple[str, Dict]:
        """同步推理接口"""
        pass

    @abstractmethod
    async def generate_async(self, messages: List[Dict[str, str]],
                           max_retries: int = 4,
                           max_tokens: int = 6000,
                           temperature: float = 0.6,
                           url: str = None,
                           timeout: float = None) -> str:
        """异步生成响应"""
        pass

    @abstractmethod
    async def infer_async(self, message: Union[str, Dict[str, str], List[Dict[str, str]]],
                         max_tokens: int = 4096,
                         history: List = None,
                         sampling_params: Dict = None,
                         temperature: float = 0.6,
                         timeout: float = None) -> Tuple[str, Dict]:
        """异步推理接口"""
        pass

    @abstractmethod
    async def infer_batch_async(self, messages: List[Union[str, Dict[str, str], List[Dict[str, str]]]],
                               max_tokens: int = 4096,
                               temperature: float = 0.6,
                               max_concurrent: int = 15,
                               output_file: str = None,
                               timeout: float = None) -> List[str]:
        """异步批量处理"""
        pass

    def format_messages(self, message: Union[str, Dict[str, str], List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        将各种格式的消息转换为标准的消息列表格式
        
        Args:
            message: 可以是字符串、单个消息字典或消息字典列表
            
        Returns:
            标准格式的消息列表
        """
        # 检查是否需要添加额外提示
        has_extra = hasattr(self, 'extra_prompt') and self.extra_prompt
        has_system = hasattr(self, 'system_prompt') and self.system_prompt
        
        result = []
        
        # 添加系统提示（如果有）
        if has_system:
            result.append({'role': 'system', 'content': self.system_prompt})
        
        if isinstance(message, dict) and 'role' in message and 'content' in message:
            # 单个消息字典
            if has_extra and message['role'] == 'user':
                message = message.copy()  # 创建副本，避免修改原始对象
                # 如果传入的massage['content']是列表，就不做拼接额外prompt的操作
                if isinstance(message['content'], list):
                    pass
                else:
                    message['content'] = f"{message['content']}\n{self.extra_prompt}"
            result.append(message)
        
        elif isinstance(message, list):
            # 消息字典列表
            if has_extra:
                for item in message:
                    item_copy = item.copy()  # 创建副本
                    if item_copy.get('role') == 'user':
                        item_copy['content'] = f"{item_copy['content']}\n{self.extra_prompt}"
                    result.append(item_copy)
            else:
                result.extend(message)
        
        else:
            # 字符串消息
            content = str(message)
            if has_extra:
                content = f"{content}\n{self.extra_prompt}"
            result.append({'role': 'user', 'content': content})
        return result