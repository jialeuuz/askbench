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
    Check whether a URL is healthy (wait up to max_wait_minutes).
    
    Args:
        url: API URL
        sk_token: auth token
        api_type: API type
        max_wait_minutes: max wait time (minutes)
        
    Returns:
        bool: whether the URL is healthy
    """
    headers = {'Content-Type': 'application/json'}
    if sk_token and sk_token != 'none':
        headers['Authorization'] = f'Bearer {sk_token}'
    
    # A minimal request payload for probing.
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "temperature": 1
    }
    
    if api_type and api_type != 'none':
        data["model"] = api_type
    
    logger.info(f"Checking URL health: {url} (max wait: {max_wait_minutes} minutes)")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    while time.time() - start_time < max_wait_seconds:
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=data, 
                timeout=30  # per-request timeout (seconds)
            )
            
            if response.status_code == 200:
                logger.info(f"URL health check passed: {url}")
                return True
            else:
                logger.warning(f"URL returned non-200: {url} (status={response.status_code})")
        
        except requests.Timeout:
            logger.warning(f"URL request timed out: {url}")
        
        except Exception as e:
            logger.warning(f"URL health check error: {url} (error={e})")
        
        # Remaining wait time
        elapsed = time.time() - start_time
        remaining = max_wait_seconds - elapsed
        
        if remaining <= 0:
            break
            
        logger.info(f"Retrying in 1 minute (remaining: {int(remaining/60)+1} minutes)")
        time.sleep(60)  # wait 1 minute between retries
    
    logger.error(f"URL health check failed after {max_wait_minutes} minutes: {url}")
    return False

async def async_check_url_health(
    url: str, 
    sk_token: str = None, 
    api_type: str = None,
    max_wait_minutes: int = 15
) -> Tuple[str, bool]:
    """
    Asynchronously check whether a URL is healthy.
    
    Args:
        url: API URL
        sk_token: auth token
        api_type: API type
        max_wait_minutes: max wait time (minutes)
        
    Returns:
        Tuple[str, bool]: (URL, is_healthy)
    """
    headers = {'Content-Type': 'application/json'}
    if sk_token and sk_token != 'none':
        headers['Authorization'] = f'Bearer {sk_token}'
    
    # A minimal request payload for probing.
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "temperature": 1
    }
    
    if api_type and api_type != 'none':
        data["model"] = api_type
    
    logger.info(f"Checking URL health: {url} (max wait: {max_wait_minutes} minutes)")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < max_wait_seconds:
            try:
                async with session.post(
                    url, 
                    headers=headers, 
                    json=data, 
                    timeout=30  # per-request timeout (seconds)
                ) as response:
                    if response.status == 200:
                        logger.info(f"URL health check passed: {url}")
                        return url, True
                    else:
                        logger.warning(f"URL returned non-200: {url} (status={response.status})")
            
            except asyncio.TimeoutError:
                logger.warning(f"URL request timed out: {url}")
            
            except Exception as e:
                logger.warning(f"URL health check error: {url} (error={e})")
            
            # Remaining wait time
            elapsed = time.time() - start_time
            remaining = max_wait_seconds - elapsed
            
            if remaining <= 0:
                break
                
            logger.info(f"Retrying in 1 minute (remaining: {int(remaining/60)+1} minutes)")
            await asyncio.sleep(60)  # wait 1 minute between retries
    
    logger.error(f"URL health check failed after {max_wait_minutes} minutes: {url}")
    return url, False

async def async_check_urls_health(
    urls: List[str], 
    sk_token: str = None, 
    api_type: str = None,
    max_wait_minutes: int = 15
) -> Tuple[bool, List[str]]:
    """
    Asynchronously check multiple URLs for health.
    
    Args:
        urls: list of API URLs
        sk_token: auth token
        api_type: API type
        max_wait_minutes: max wait time (minutes)
        
    Returns:
        Tuple[bool, List[str]]: (has_healthy_url, healthy_url_list)
    """
    if not urls:
        logger.error("URL list is empty")
        return False, []
    
    # Probe all URLs concurrently
    tasks = [
        async_check_url_health(url, sk_token, api_type, max_wait_minutes) 
        for url in urls
    ]
    results = await asyncio.gather(*tasks)
    
    # Extract healthy URLs
    healthy_urls = [url for url, is_healthy in results if is_healthy]
    
    success = len(healthy_urls) > 0
    
    if not success:
        logger.error(f"All URLs are unhealthy ({len(urls)} total)")
    else:
        logger.info(f"Found {len(healthy_urls)}/{len(urls)} healthy URLs")
    
    return success, healthy_urls

def check_urls_health(
    urls: List[str], 
    sk_token: str = None, 
    api_type: str = None,
    max_wait_minutes: int = 15
) -> Tuple[bool, List[str]]:
    """
    Check multiple URLs for health (sync wrapper).
    
    Args:
        urls: list of API URLs
        sk_token: auth token
        api_type: API type
        max_wait_minutes: max wait time (minutes)
        
    Returns:
        Tuple[bool, List[str]]: (has_healthy_url, healthy_url_list)
    """
    # Detect whether we're already running inside an event loop.
    try:
        # If we are already in an event loop, run the async probe in a separate thread.
        loop = asyncio.get_running_loop()
        
        # Create a new event loop in a new thread to run the async tasks.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                lambda: asyncio.run(
                    async_check_urls_health(urls, sk_token, api_type, max_wait_minutes)
                )
            )
            return future.result()
            
    except RuntimeError:
        # If not in an event loop, we can use asyncio.run directly.
        return asyncio.run(
            async_check_urls_health(urls, sk_token, api_type, max_wait_minutes)
        )
