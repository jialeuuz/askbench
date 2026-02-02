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

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Globals for active models
_active_models = []
_keepalive_thread = None
_stop_keepalive = False

KEEPALIVE_PROMPT = (
    "Hello! This is a lightweight keepalive ping. Please respond with a short acknowledgement."
)
KEEPALIVE_MESSAGES = [{"role": "user", "content": KEEPALIVE_PROMPT}]

def _keepalive_worker(interval_minutes=1):
    """
    Background worker that periodically sends keepalive requests to all URLs of active models.
    
    Args:
        interval_minutes: keepalive interval (minutes)
    """
    global _stop_keepalive
    # logger.info(f"Keepalive thread started; interval: {interval_minutes} minutes")
    
    while not _stop_keepalive:
        # Wait for the configured interval
        for _ in range(interval_minutes * 60):
            if _stop_keepalive:
                break
            time.sleep(1)
            
        if _stop_keepalive:
            break
            
        # Send keepalive requests for each active model and URL
        for model in _active_models:
            # If the model has multiple URLs (api_urls), ping each one.
            if hasattr(model, 'api_urls') and len(model.api_urls) > 1:
                for url in model.api_urls:
                    try:
                        # For models that support async generation, try generating with the specified URL.
                        if hasattr(model, 'generate_async'):
                            # Create an event loop for async execution.
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
                            # logger.info(f"Keepalive sent to URL: {url}, model: {model.__class__.__name__}")
                        else:
                            # If the model cannot target a specific URL in generate(), warn and continue.
                            logger.warning(f"Model {model.__class__.__name__} does not support per-URL keepalive: {url}")
                    except Exception as e:
                        pass
                        # logger.warning(f"URL keepalive failed: {url}, error: {e}")
            else:
                try:
                    response = model.generate(
                        messages=KEEPALIVE_MESSAGES,
                        max_tokens=16384,
                        temperature=1
                    )
                    # logger.info(f"Keepalive sent to primary URL: {model.url}, model: {model.__class__.__name__}")
                except Exception as e:
                    pass
                    # logger.warning(f"Primary URL keepalive failed: {model.url}, error: {e}")
    
    # logger.info("Keepalive thread stopped")

def start_keepalive_thread():
    """Start the keepalive thread (if not already running)."""
    global _keepalive_thread, _stop_keepalive
    
    if _keepalive_thread is None or not _keepalive_thread.is_alive():
        _stop_keepalive = False
        _keepalive_thread = threading.Thread(target=_keepalive_worker, daemon=True)
        _keepalive_thread.start()

def register_model_for_keepalive(model: BaseAPIModel):
    """
    Register a model instance for keepalive.
    
    Args:
        model: the model instance to keep alive
    """
    global _active_models
    
    if model not in _active_models:
        _active_models.append(model)
        logger.info(f"Registered model for keepalive: {model.__class__.__name__}")
    
    # Ensure the keepalive thread is running.
    start_keepalive_thread()

def create_model(model_config: Dict, generate_config: Dict = None) -> BaseAPIModel:
    """
    Create a model instance from config.
    
    Args:
        model_config: model config dict
        
    Returns:
        The created model instance
    """
    # Handle multi-URL config
    if "api_url" in model_config and "," in model_config["api_url"]:
        api_urls = [url.strip() for url in model_config["api_url"].split(",")]
        main_url = api_urls[0]  # first URL is the primary URL
        logger.info(f"Detected {len(api_urls)} API URLs in config")
    else:
        api_urls = None
        main_url = model_config.get("api_url")
    
    # Timeout (default: 600s)
    timeout = model_config.get("timeout", 600)
    if isinstance(timeout, str):
        timeout = float(timeout)
    logger.info(f"Using timeout: {timeout} seconds")
    
    # Extra prompt
    extra_prompt = model_config.get("extra_prompt")
    if extra_prompt:
        logger.info(f"Using extra_prompt: {extra_prompt}")
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
        else:  # default: custom
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
        raise ValueError(f"Unsupported model type: {model_config.get('model_type')}")
    
    # Health check
    logger.info("Checking API URL health...")
    print('model:\n\n')
    print(model)
    is_healthy = model.check_health()
    
    if not is_healthy:
        logger.error("All API URLs are unhealthy; exiting")
        raise ValueError("All API URLs are unhealthy; exiting")
    
    if 'fc.chj.cloud' in model.url:
        return model
    else:
        # Register model for keepalive
        # register_model_for_keepalive(model)
        return model
