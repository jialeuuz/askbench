import time
import requests
import logging
import asyncio
import aiohttp
from typing import List, Tuple

logger = logging.getLogger(__name__)

def check_url_health(
    url: str, 
    sk_token: str = None, 
    api_type: str = None,
    max_wait_minutes: int = 15
) -> bool:
    """
    检查URL的健康状态，最多等待15分钟
    
    Args:
        url: API URL
        sk_token: 授权令牌
        api_type: API类型
        max_wait_minutes: 最长等待时间（分钟）
        
    Returns:
        bool: URL是否健康
    """
    headers = {'Content-Type': 'application/json'}
    if sk_token and sk_token != 'none':
        headers['Authorization'] = f'Bearer {sk_token}'
    
    # 构造一个简单的请求数据
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "temperature": 1
    }
    
    if api_type and api_type != 'none':
        data["model"] = api_type
    
    logger.info(f"开始检测URL健康状态: {url}, 最多等待{max_wait_minutes}分钟")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    while time.time() - start_time < max_wait_seconds:
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=data, 
                timeout=30  # 单次请求30秒超时
            )
            
            if response.status_code == 200:
                logger.info(f"URL健康检测成功: {url}")
                return True
            else:
                logger.warning(f"URL响应异常: {url}, 状态码: {response.status_code}")
        
        except requests.Timeout:
            logger.warning(f"URL请求超时: {url}")
        
        except Exception as e:
            logger.warning(f"URL检测异常: {url}, 错误: {e}")
        
        # 计算剩余等待时间
        elapsed = time.time() - start_time
        remaining = max_wait_seconds - elapsed
        
        if remaining <= 0:
            break
            
        logger.info(f"将在1分钟后重试, 剩余等待时间: {int(remaining/60)+1}分钟")
        time.sleep(60)  # 每次等待1分钟后重试
    
    logger.error(f"URL健康检测失败，已等待{max_wait_minutes}分钟: {url}")
    return False

async def async_check_url_health(
    url: str, 
    sk_token: str = None, 
    api_type: str = None,
    max_wait_minutes: int = 15
) -> Tuple[str, bool]:
    """
    异步检查URL的健康状态
    
    Args:
        url: API URL
        sk_token: 授权令牌
        api_type: API类型
        max_wait_minutes: 最长等待时间（分钟）
        
    Returns:
        Tuple[str, bool]: (URL, 是否健康)
    """
    headers = {'Content-Type': 'application/json'}
    if sk_token and sk_token != 'none':
        headers['Authorization'] = f'Bearer {sk_token}'
    
    # 构造一个简单的请求数据
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "temperature": 1
    }
    
    if api_type and api_type != 'none':
        data["model"] = api_type
    
    logger.info(f"开始检测URL健康状态: {url}, 最多等待{max_wait_minutes}分钟")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < max_wait_seconds:
            try:
                async with session.post(
                    url, 
                    headers=headers, 
                    json=data, 
                    timeout=30  # 单次请求30秒超时
                ) as response:
                    if response.status == 200:
                        logger.info(f"URL健康检测成功: {url}")
                        return url, True
                    else:
                        logger.warning(f"URL响应异常: {url}, 状态码: {response.status}")
            
            except asyncio.TimeoutError:
                logger.warning(f"URL请求超时: {url}")
            
            except Exception as e:
                logger.warning(f"URL检测异常: {url}, 错误: {e}")
            
            # 计算剩余等待时间
            elapsed = time.time() - start_time
            remaining = max_wait_seconds - elapsed
            
            if remaining <= 0:
                break
                
            logger.info(f"将在1分钟后重试, 剩余等待时间: {int(remaining/60)+1}分钟")
            await asyncio.sleep(60)  # 每次等待1分钟后重试
    
    logger.error(f"URL健康检测失败，已等待{max_wait_minutes}分钟: {url}")
    return url, False

async def async_check_urls_health(
    urls: List[str], 
    sk_token: str = None, 
    api_type: str = None,
    max_wait_minutes: int = 15
) -> Tuple[bool, List[str]]:
    """
    异步检查多个URL的健康状态
    
    Args:
        urls: API URL列表
        sk_token: 授权令牌
        api_type: API类型
        max_wait_minutes: 最长等待时间（分钟）
        
    Returns:
        Tuple[bool, List[str]]: (是否有健康的URL, 健康URL列表)
    """
    if not urls:
        logger.error("URL列表为空")
        return False, []
    
    # 并发检查所有URL
    tasks = [
        async_check_url_health(url, sk_token, api_type, max_wait_minutes) 
        for url in urls
    ]
    results = await asyncio.gather(*tasks)
    
    # 从结果中提取健康的URL
    healthy_urls = [url for url, is_healthy in results if is_healthy]
    
    success = len(healthy_urls) > 0
    
    if not success:
        logger.error(f"所有URL({len(urls)}个)都不健康")
    else:
        logger.info(f"找到{len(healthy_urls)}/{len(urls)}个健康的URL")
    
    return success, healthy_urls

def check_urls_health(
    urls: List[str], 
    sk_token: str = None, 
    api_type: str = None,
    max_wait_minutes: int = 15
) -> Tuple[bool, List[str]]:
    """
    检查多个URL的健康状态 (同步包装器)
    
    Args:
        urls: API URL列表
        sk_token: 授权令牌
        api_type: API类型
        max_wait_minutes: 最长等待时间（分钟）
        
    Returns:
        Tuple[bool, List[str]]: (是否有健康的URL, 健康URL列表)
    """
    # 检查当前是否在事件循环中运行
    try:
        # 如果已经在事件循环中，则直接使用asyncio.run_coroutine_threadsafe或通过其他方式执行
        loop = asyncio.get_running_loop()
        
        # 创建一个新的事件循环在新线程中运行异步任务
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                lambda: asyncio.run(
                    async_check_urls_health(urls, sk_token, api_type, max_wait_minutes)
                )
            )
            return future.result()
            
    except RuntimeError:
        # 如果不在事件循环中，则可以直接使用asyncio.run
        return asyncio.run(
            async_check_urls_health(urls, sk_token, api_type, max_wait_minutes)
        )