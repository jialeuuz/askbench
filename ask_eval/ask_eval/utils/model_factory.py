# ask_eval/utils/model_factory.py
import sys
import threading
import time
from typing import Dict, List
import logging
from ask_eval.models.base_api_model import BaseAPIModel
from ask_eval.models.api.custom import CustomAPI
from ask_eval.models.api.gpt import GPTAPI
import asyncio

# 配置日志输出到终端
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 存储活跃模型的全局变量
_active_models = []
_keepalive_thread = None
_stop_keepalive = False

KEEPALIVE_PROMPT = (
    "Hello, this is a keepalive message but please respond strictly in Chinese. "
    "已知 $f ( x )=2 \\operatorname{s i n} \\frac{\\pi x} {3}, g ( x )=\\frac{1} {x-6}$，"
    "则 $f ( x )=g ( x )$ 在 $[-8, 2 0 ]$ 上所有根的和为多少？"
)
KEEPALIVE_MESSAGES = [{"role": "user", "content": KEEPALIVE_PROMPT}]

def _keepalive_worker(interval_minutes=1):
    """
    后台工作线程，定期为所有活跃模型的所有URL发送保活请求
    
    Args:
        interval_minutes: 保活间隔时间（分钟）
    """
    global _stop_keepalive
    # logger.info(f"模型保活线程已启动，间隔: {interval_minutes}分钟")
    
    while not _stop_keepalive:
        # 等待指定的间隔时间
        for _ in range(interval_minutes * 60):
            if _stop_keepalive:
                break
            time.sleep(1)
            
        if _stop_keepalive:
            break
            
        # 为每个活跃模型的每个URL发送保活请求
        for model in _active_models:
            # 如果模型有多个URL(api_urls列表)，遍历每个URL
            if hasattr(model, 'api_urls') and len(model.api_urls) > 1:
                for url in model.api_urls:
                    try:
                        # 对于DeepSeekAPI和其他支持异步生成的模型，尝试使用指定URL生成
                        if hasattr(model, 'generate_async'):
                            # 这里需要添加异步运行上下文
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(model.generate_async(
                                messages=KEEPALIVE_MESSAGES,
                                max_tokens=16384,
                                temperature=1,
                                url=url,
                                timeout=360
                            ))
                            loop.close()
                            # logger.info(f"已发送保活请求到URL: {url}, 模型: {model.__class__.__name__}")
                        else:
                            # 对于不支持在generate方法中指定URL的模型，记录警告
                            logger.warning(f"模型 {model.__class__.__name__} 不支持对URL进行保活: {url}")
                    except Exception as e:
                        pass
                        # logger.warning(f"URL保活请求失败: {url}, 错误: {e}")
            else:
                try:
                    response = model.generate(
                        messages=KEEPALIVE_MESSAGES,
                        max_tokens=16384,
                        temperature=1
                    )
                    # logger.info(f"已发送保活请求到主URL: {model.url}, 模型: {model.__class__.__name__}")
                except Exception as e:
                    pass
                    # logger.warning(f"主URL保活请求失败: {model.url}, 错误: {e}")
    
    # logger.info("模型保活线程已停止")

def start_keepalive_thread():
    """启动保活线程（如果尚未启动）"""
    global _keepalive_thread, _stop_keepalive
    
    if _keepalive_thread is None or not _keepalive_thread.is_alive():
        _stop_keepalive = False
        _keepalive_thread = threading.Thread(target=_keepalive_worker, daemon=True)
        _keepalive_thread.start()

def register_model_for_keepalive(model: BaseAPIModel):
    """
    注册模型以进行保活
    
    Args:
        model: 需要保活的模型实例
    """
    global _active_models
    
    if model not in _active_models:
        _active_models.append(model)
        logger.info(f"已注册模型进行保活: {model.__class__.__name__}")
    
    # 确保保活线程已启动
    start_keepalive_thread()

def create_model(model_config: Dict, generate_config: Dict = None) -> BaseAPIModel:
    """
    根据配置创建模型实例
    
    Args:
        model_config: 模型配置字典
        
    Returns:
        创建的模型实例
    """
    # 处理可能存在的多个URL
    if "api_url" in model_config and "," in model_config["api_url"]:
        api_urls = [url.strip() for url in model_config["api_url"].split(",")]
        main_url = api_urls[0]  # 第一个URL作为主URL
        logger.info(f"检测到多个URL配置，共 {len(api_urls)} 个URL")
    else:
        api_urls = None
        main_url = model_config.get("api_url")
    
    # 获取timeout参数，默认为600秒
    timeout = model_config.get("timeout", 600)
    if isinstance(timeout, str):
        timeout = float(timeout)
    logger.info(f"使用超时设置: {timeout}秒")
    
    # 获取额外提示
    extra_prompt = model_config.get("extra_prompt")
    if extra_prompt:
        logger.info(f"使用额外提示: {extra_prompt}")
    if model_config.get("model_type") == "api":
        api_type_value = model_config.get("api_type", "")
        api_type = api_type_value.lower()
        model_name_value = model_config.get("model_name") or ""
        model_name_lower = model_name_value.lower()
        use_gpt_api = False

        if "gpt" in api_type:
            use_gpt_api = True
        elif model_name_lower and model_name_lower != "default":
            use_gpt_api = True

        if use_gpt_api:
            model = GPTAPI(
                url=main_url,
                api_type=api_type_value,
                model_name=model_config.get("model_name"),
                sk_token=model_config.get("sk_token", "none"),
                api_urls=api_urls,
                timeout=timeout,
                extra_prompt=extra_prompt,
                system_prompt=model_config.get("system_prompt"),
                generate_config=generate_config
            )
        else:  # 默认为 custom
            model = CustomAPI(
                url=main_url,
                sk_token=model_config.get("sk_token", "none"),
                api_type=model_config.get("api_type", "custom-reasoner"),
                api_urls=api_urls,
                timeout=timeout,
                extra_prompt=extra_prompt,
                system_prompt=model_config.get("system_prompt"),
                generate_config=generate_config
            )
    else:
        print(model_config)
        raise ValueError(f"不支持的模型类型: {model_config.get('model_type')}")
    
    # 检查模型健康状态
    logger.info(f"正在检查API URL健康状态...")
    print('model:\n\n')
    print(model)
    is_healthy = model.check_health()
    
    if not is_healthy:
        logger.error("所有API URL都不健康，程序退出")
        raise ValueError(f"所有API URL都不健康，程序退出")
    
    if 'fc.chj.cloud' in model.url:
        return model
    else:
        # 注册模型进行保活
        # register_model_for_keepalive(model)
        return model
