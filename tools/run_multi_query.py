import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from asyncio import Semaphore
from datetime import datetime
from tqdm import tqdm
import requests
import asyncio
import aiohttp
import base64
import json
import time
from typing import List, Union
import threading
import traceback

# 异步限流器 (用于 asyncio 模式)
class RateLimiter:
    def __init__(self, rate):
        self._interval = 1.0 / rate
        self._last_time = None

    async def wait(self):
        now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            return
        elapsed = now - self._last_time
        if elapsed < self._interval:
            await asyncio.sleep(self._interval - elapsed)
        self._last_time = time.monotonic()

# 同步限流器 (用于多线程模式)
class SyncRateLimiter:
    def __init__(self, rate):
        self._interval = 1.0 / rate
        self._last_time = None
        self._lock = threading.Lock() # 确保在多线程环境下的原子性操作

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if self._last_time is None:
                self._last_time = now
                return
            
            elapsed = now - self._last_time
            sleep_for = self._interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_time = time.monotonic()

class ZnyConfig:
    def __init__(self, url: Union[str, List[str]], model_name: str = 'gpt4o', temperature: float = 0.9, max_retries: int = 5, retry_until_success: bool = False, 
                 qps: int = 2, # 此处的QPS现在代表 *每个API端点* 的上限
                 max_concurrent: int = 10, chunk_size: int = 1, asyncio_flag: bool = False, image_flag: bool = False, image_column_name: str = None, input_column_name: str = 'input', response_column_name: str = "assistant",
                 top_p=0.95, repetition_penalty=1, max_tokens=6400, 
                 resume_from_output: bool = False,
                 save_interval: int = 100):
        
        if isinstance(url, str):
            self.urls = [u.strip() for u in url.split(',') if u.strip()]
        elif isinstance(url, list):
            self.urls = [u.strip() for u in url if u.strip()]
        else:
            raise ValueError("url must be a string (comma-separated) or a list of strings.")
        
        if not self.urls:
            raise ValueError("No valid URLs provided.")

        self.model_name = model_name
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.retry_until_success = retry_until_success
        self.temperature = temperature
        self.qps = qps
        self.max_concurrent = max_concurrent
        self.asyncio_flag = asyncio_flag
        self.image_flag = image_flag
        self.image_column_name = image_column_name
        self.input_column_name = input_column_name
        self.response_column_name = response_column_name
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.resume_from_output = resume_from_output
        self.save_interval = save_interval

class CallLLMByZny(object):
    def __init__(self, config: ZnyConfig):
        self.config = config
        self.url_list = self.config.urls
        
        self._url_index = 0
        self._url_lock = threading.Lock()
        
        self.rate_limiters = {}
        if config.asyncio_flag:
            for url in self.url_list:
                self.rate_limiters[url] = RateLimiter(config.qps)
        else:
            for url in self.url_list:
                self.rate_limiters[url] = SyncRateLimiter(config.qps)

        print(f"🚀 LLM服务已初始化，将使用以下API端点进行负载均衡: {self.url_list}")
        print(f"⚡️ 限流策略: 每个API端点的QPS上限为 {config.qps} (总并发连接数上限为 {config.max_concurrent})")

    def _get_next_url(self) -> str:
        """线程安全地获取下一个URL，实现轮询。"""
        with self._url_lock:
            url = self.url_list[self._url_index]
            self._url_index = (self._url_index + 1) % len(self.url_list)
            return url

    def _save_progress(self, df_to_append: pd.DataFrame, out_path: str):
        if df_to_append.empty:
            return
        try:
            jsonl_string = df_to_append.to_json(
                orient="records", lines=True, force_ascii=False
            )
            if not jsonl_string.endswith("\n"):
                jsonl_string += "\n"
            
            with open(out_path, 'a', encoding='utf-8') as f:
                f.write(jsonl_string)
            print(f"✔ 进度已保存：成功追加 {len(df_to_append)} 条记录到 {out_path}")
        except Exception as e:
            print(f"❌ 保存进度时出错: {e}")

    def _read_jsonl_robustly(self, file_path: str) -> pd.DataFrame:
        valid_records = []
        corrupted_lines_count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        valid_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        corrupted_lines_count += 1
                        print(f"  - 警告: 在文件 {os.path.basename(file_path)} 中检测到并跳过第 {i+1} 行的损坏数据。")
            
            if corrupted_lines_count > 0:
                print(f"  - 总计: 从 {os.path.basename(file_path)} 中成功加载 {len(valid_records)} 条记录，忽略了 {corrupted_lines_count} 条损坏的记录。")

            if not valid_records:
                return pd.DataFrame()
            
            return pd.DataFrame(valid_records)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ 读取文件 {file_path} 时发生未知错误: {e}")
            return pd.DataFrame()

    def get_gpt4api_df(self, init_prompt_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
        prompt_df = init_prompt_df.copy()
        out_name = os.path.basename(out_path)
        
        processed_inputs = set()
        if self.config.resume_from_output and os.path.exists(out_path):
            print(f"检测到输出文件 {out_path}，正在尝试断点重续...")
            processed_df = self._read_jsonl_robustly(out_path)
            
            if not processed_df.empty:
                if self.config.input_column_name in processed_df.columns:
                    processed_inputs = set(processed_df[self.config.input_column_name])
                    print(f"已从 {out_path} 加载 {len(processed_inputs)} 条有效记录。")
                else:
                    print(f"警告：输出文件中缺少输入列 '{self.config.input_column_name}'，无法进行精确的断点重续。将重新处理所有数据。")
            else:
                print("输出文件为空或所有行均已损坏。将从头开始处理。")

        df_to_process = prompt_df[~prompt_df[self.config.input_column_name].isin(processed_inputs)].copy()
        
        if df_to_process.empty:
            print("所有数据均已处理完毕。")
            return self._read_jsonl_robustly(out_path) if os.path.exists(out_path) else pd.DataFrame()

        print(f"总计 {len(prompt_df)} 条，已处理 {len(processed_inputs)} 条，本次需处理 {len(df_to_process)} 条。")
        df_to_process.reset_index(drop=True, inplace=True)
        
        all_prompts_ls = list(zip(
            df_to_process.index,
            df_to_process[self.config.input_column_name].to_list(),
            df_to_process[self.config.image_column_name].to_list() if self.config.image_column_name and self.config.image_column_name in df_to_process.columns else [None] * len(df_to_process)
        ))

        if self.config.asyncio_flag:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(self._process_and_save_stream(all_prompts_ls, df_to_process, out_path, out_name))
        else:
            self._process_and_save_stream_threaded(all_prompts_ls, df_to_process, out_path, out_name)

        print(f"所有任务处理完成。正在读取最终结果文件...")
        if os.path.exists(out_path):
            return self._read_jsonl_robustly(out_path)
        else:
            return pd.DataFrame()

    def _process_and_save_stream_threaded(self, indexed_prompts, source_df, out_path, out_name):
        newly_completed_rows = []
        with tqdm(total=len(indexed_prompts), desc=f"{out_name}进度") as pbar:
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
                futures = {executor.submit(self._process_one_prompt_with_index, index, [prompt], image_path): index 
                           for index, prompt, image_path in indexed_prompts}

                for future in as_completed(futures):
                    try:
                        original_index, response_data = future.result()
                        response_content = self._parser_one_response(response_data)
                        
                        if "<|ERROR" in response_content or "<|PARSING_ERROR|>" in response_content:
                            print(f"  -> 任务 {original_index} 失败，已跳过保存。错误: {response_content}")
                            continue

                        row_data = source_df.loc[original_index].to_dict()
                        row_data[self.config.response_column_name] = response_content
                        newly_completed_rows.append(row_data)
                        
                        if self.config.save_interval > 0 and len(newly_completed_rows) >= self.config.save_interval:
                            self._save_progress(pd.DataFrame(newly_completed_rows), out_path)
                            newly_completed_rows = []

                    except Exception as e:
                        print(f"处理任务 {original_index} 时发生严重错误: {repr(e)}")
                        traceback.print_exc()
                    finally:
                        pbar.update(1)
        
        if newly_completed_rows:
            self._save_progress(pd.DataFrame(newly_completed_rows), out_path)

    async def _process_prompt_with_retries(self, semaphore, session, index, prompt, image_path, max_retries=5):
        for attempt in range(1, max_retries + 1):
            try:
                # 执行一次实际调用
                original_index, response_data = await self._process_one_prompt_async_with_index(
                    semaphore, session, index, [prompt], image_path
                )
                response_content = self._parser_one_response(response_data)

                # 判断返回值是否包含错误标记
                if "<|ERROR" in response_content or "<|PARSING_ERROR|>" in response_content:
                    print(f"任务 {original_index} 第 {attempt} 次返回错误，准备重试...")
                    await asyncio.sleep(1)  # 可选: 等待1s避免连环错误
                    continue

                # 成功
                return True, original_index, response_content

            except Exception as e:
                print(f"处理任务 {index} 第 {attempt} 次发生异常: {repr(e)}")
                traceback.print_exc()
                await asyncio.sleep(1)  # 避免频繁重试

        # 如果到了这里，表示5次都失败
        return False, index, None

    async def _process_and_save_stream(self, indexed_prompts, source_df, out_path, out_name):
        semaphore = Semaphore(self.config.max_concurrent)
        newly_completed_rows = []

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._process_prompt_with_retries(
                    semaphore, session, index, prompt, image_path, max_retries=5
                )
                for index, prompt, image_path in indexed_prompts
            ]

            with tqdm(total=len(tasks), desc=f"{out_name}进度") as pbar:
                for future in asyncio.as_completed(tasks):
                    success, original_index, response_content = await future

                    if success:
                        row_data = source_df.loc[original_index].to_dict()
                        row_data[self.config.response_column_name] = response_content
                        newly_completed_rows.append(row_data)

                        if self.config.save_interval > 0 and len(newly_completed_rows) >= self.config.save_interval:
                            self._save_progress(pd.DataFrame(newly_completed_rows), out_path)
                            newly_completed_rows = []
                    else:
                        print(f"任务 {original_index} 重试 {5} 次依旧失败，已跳过。")

                    pbar.update(1)

        if newly_completed_rows:
            self._save_progress(pd.DataFrame(newly_completed_rows), out_path)

    def _process_one_prompt_with_index(self, index: int, prompt: list, image_path: str = None) -> tuple:
        response = self._request_one_chat(prompt, image_path)
        return index, response

    async def _process_one_prompt_async_with_index(self, semaphore, session, index: int, prompts: list, image_path: str) -> tuple:
        response = await self._request_one_chat_async(semaphore, session, prompts, image_path)
        return index, response

    def _request_one_chat(self, messages: list, image_path: str):
        headers = {'Content-Type': 'application/json'}
        # <<< 改动 2.1: 为 "default" 模型自动添加认证头
        if self.config.model_name == 'default':
            headers['Authorization'] = 'Bearer EMPTY'
            
        data_entry = self._make_chat_request_entry(messages, image_path)
        retries = 0
        last_exception = None

        while True:
            url = self._get_next_url()
            limiter = self.rate_limiters[url]
            
            limiter.wait()
            try:
                response = requests.post(url, headers=headers, json=data_entry, timeout=60)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_exception = e
                print(f"  - 请求URL失败: {url}. 错误: {e}. 将在下一次尝试中使用下一个URL。")

            retries += 1
            if not self.config.retry_until_success and retries >= self.config.max_retries:
                error_msg = f"已达最大重试次数({self.config.max_retries})，所有尝试均失败。最终错误: {last_exception}"
                print(f"❌ {error_msg}")
                return {"error": error_msg}
            
            retry_msg = f"无限重试... (尝试次数: {retries})" if self.config.retry_until_success else f"重试... (尝试 {retries}/{self.config.max_retries})"
            print(f"  - {retry_msg}")
            time.sleep(2)

    async def _request_one_chat_async(self, semaphore, session, messages, image_path):
        headers = {'Content-Type': 'application/json'}
        if self.config.model_name == 'default':
            headers['Authorization'] = 'Bearer EMPTY'

        data_entry = self._make_chat_request_entry(messages, image_path)
        retries = 0
        last_exception = None

        # 这里用 aiohttp.ClientTimeout 可以细分连接、读取、总超时
        timeout_cfg = aiohttp.ClientTimeout(
            total=120,     # 整个请求最长时间
            connect=10,    # TCP 连接阶段超时
            sock_read=800  # 等待服务器响应数据的最长时间
        )

        while True:
            url = self._get_next_url()
            limiter = self.rate_limiters[url]

            try:
                async with semaphore:
                    await limiter.wait()
                    async with session.post(url, json=data_entry, headers=headers, timeout=timeout_cfg) as response:
                        response.raise_for_status()
                        return await response.json()

            except asyncio.TimeoutError as e:
                # ⏳ 明确打印超时，并进入重试逻辑
                print(f"⚠️ 请求超时：{url} (第 {retries+1} 次尝试，共 {self.config.max_retries} 次)")
                # traceback.print_exc()
                last_exception = e

            except aiohttp.ClientError as e:
                # aiohttp 连接类错误（ConnectionResetError, ServerDisconnectedError等）
                print(f"❌ 网络请求失败：{url} 错误类型: {type(e).__name__}, 信息: {e}")
                # traceback.print_exc()
                last_exception = e

            except Exception as e:
                # 其他未知错误
                print(f"❌ 未知异常：{url} 错误类型: {type(e).__name__}, 信息: {e}")
                traceback.print_exc()
                last_exception = e

            # =========================
            # 统一的重试判定与等待逻辑
            # =========================
            retries += 1
            if not self.config.retry_until_success and retries >= self.config.max_retries:
                error_msg = f"已达最大重试次数({self.config.max_retries})，最后错误类型: {type(last_exception).__name__}, 信息: {last_exception}"
                print(f"❌ {error_msg}")
                return {"error": error_msg}

            retry_msg = (
                f"无限重试（当前已尝试 {retries} 次）..."
                if self.config.retry_until_success
                else f"重试中... (尝试 {retries}/{self.config.max_retries})"
            )
            print(f"  - {retry_msg}")

            # 可以改成指数退避（exponential backoff）降低压力
            await asyncio.sleep(min(2 * retries, 10))

    def encode_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _make_chat_request_entry(self, messages: list, image_path) -> dict:
        prompt_text = messages[0]

        if self.config.model_name == "gpt4o":
            if self.config.image_flag:
                assert image_path is not None, "image_flag = True 而 image path 为空，无法解码"
                try:
                    base64_image = self.encode_image(image_path)
                except Exception as e:
                    raise IOError(f"无法读取或编码图片: {image_path}") from e
                data_entry = {
                    "messages": [{"role": "user", "contents": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}, {"type": "text", "text": prompt_text}]}],
                    "temperature": self.config.temperature
                }
            else:
                data_entry = {
                    "messages": [{'role' : 'user', 'contents' : [{"type": "text","text": prompt_text}]}],
                    "temperature": self.config.temperature
                }
        elif self.config.model_name in ["gpt4", "wenxin"] or "claude" in self.config.model_name:
            data_entry = {
                "messages": [{"role": "user", "content": msg} for msg in messages],
                "temperature": self.config.temperature
            }
        elif self.config.model_name == 'default':
            data_entry = {
                "model": "default",
                "messages": [{"role": "user", "content": msg} for msg in messages],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
        elif "deepseek" in self.config.model_name:
            data_entry = {
                "messages": [{"role": "user", "content": msg} for msg in messages],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
                "max_tokens": self.config.max_tokens
            }
        elif any(m in self.config.model_name for m in ["o1-mini", "o4-mini", "gpt_41", "gemini_2_5_pro", "gpt_5"]):
            return {
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
            }
        else:
            data_entry = {
                "messages": [{"role": "user", "content": msg} for msg in messages],
                "temperature": self.config.temperature
            }

        return data_entry

    def _parser_one_response(self, response_item: dict):
        try:
            if 'error' in response_item:
                return f"<|ERROR: {response_item['error']}|>"
            
            if self.config.model_name == 'gpt4o':
                return response_item['data']['choices'][0]['content']
            elif self.config.model_name == 'wenxin':
                return response_item['data']['result']
            # <<< 改动 4: 将 "default" 添加到此解析逻辑中
            elif self.config.model_name in ['claude', 'deepseek_v3', 'gpt_5', 'default'] or any(m in self.config.model_name for m in ["o1-mini", "o4-mini", "gpt_41", "gemini_2_5_pro"]):
                return response_item['choices'][0]['message']['content']
            elif self.config.model_name == 'deepseek_r1':
                assistant1 = response_item['choices'][0]['message']['content']
                assistant_reasoning = response_item['choices'][0]['message']['reasoning_content']
                return json.dumps({'response': assistant1, 'reasoning': assistant_reasoning}, ensure_ascii=False)
            else:
                return response_item['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as e:
            return f"<|PARSING_ERROR: {e} - Response: {str(response_item)[:200]}|>"

def read_data(file: Union[str, list, dict]):
    def read_one_file(file_path: str, rate: float = 1.0) -> pd.DataFrame:
        print(f"正在读取文件: {file_path} ...")
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.jsonl'):
            records = []
            corrupted_lines_count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            corrupted_lines_count += 1
                            print(f"⚠️ 警告: 在输入文件 {os.path.basename(file_path)} 的第 {i+1} 行解析JSON时出错。已跳过此行。")
                            print(f"   错误信息: {e}")
                            print(f"   问题行内容 (前150字符): {line[:150]}")
                
                if corrupted_lines_count > 0:
                    print(f"ℹ️ 总计: 从 {os.path.basename(file_path)} 成功加载 {len(records)} 条记录，忽略了 {corrupted_lines_count} 条损坏的记录。")

                if not records:
                    df = pd.DataFrame()
                else:
                    df = pd.DataFrame(records)

            except FileNotFoundError:
                print(f"❌ 错误: 输入文件未找到 {file_path}")
                return pd.DataFrame()
            except Exception as e:
                print(f"❌ 读取文件 {file_path} 时发生未知错误: {e}")
                return pd.DataFrame()

        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, orient="records")
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
            
        if not df.empty:
            if rate > 1:
                n_samples = int(rate * len(df))
                df = df.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
            elif 0 < rate < 1:
                df = df.sample(frac=rate, random_state=42).reset_index(drop=True)
        
        print(f"# {os.path.basename(file_path)}: 原始 {len(records) if 'records' in locals() else len(df)} 条, 加载并采样后数据量为 {len(df)}")
        return df

    if isinstance(file, str):
        df = read_one_file(file)
    elif isinstance(file, list):
        df = pd.concat([read_one_file(path) for path in file], ignore_index=True)
        print(f"合并后总数据量: {len(df)}")
    elif isinstance(file, dict):
        assert 'rate' in file and 'path' in file, '字典格式必须包含 `rate` 和 `path` 键'
        df = pd.concat([read_one_file(path, rate=file['rate'][i]) for i, path in enumerate(file['path'])], ignore_index=True)
        print(f"合并后总数据量: {len(df)}")
    else:
        raise TypeError(f"不支持的输入类型: {type(file)}")
    
    return df

def fill_prompt_by_key_mappings(df, template: str, key_mappings: dict, prompt_key: str = "prompt") -> pd.DataFrame:
    # ... 此函数无需修改，保持原样 ...
    filled_inputs = []
    for _, row in df.iterrows():
        filled_template = template
        for placeholder, real_key in key_mappings.items():
            filler = str(row.get(real_key, ''))
            filled_template = filled_template.replace(f'{{{placeholder}}}', filler)
        filled_inputs.append(filled_template)

    df[prompt_key] = filled_inputs
    return df

# <<< 改动 1: 在字典中添加您的API
ZNY_API_URLS = {
    "gpt4o": "https://yangshuling-gpt4o.fc.chj.cloud/gpt4o/chat",
    "claude": "https://yangshuling-claude.fc.chj.cloud/claude35_sonnet/conversation",
    "o1-mini": "https://yangshuling-deepseek.fc.chj.cloud/o1-mini",
    "o4-mini": "https://yangshuling-deepseek.fc.chj.cloud/o4-mini",
    "claude-37": "https://yangshuling-deepseek.fc.chj.cloud/claude-37",
    "deepseek_r1": "https://yangshuling-deepseek.fc.chj.cloud/deepseek_r1",
    "gpt_41": "https://gpt41.fc.chj.cloud/gpt_41,https://yangjingwen.fc.chj.cloud/gpt_41,https://jiale-de-deepseek.fc.chj.cloud/gpt_41",
    "gpt_5": "https://yangjingwen.fc.chj.cloud/gpt_5, https://jiale-de-deepseek.fc.chj.cloud/gpt_5,https://linzhiyu-gemini.fc.chj.cloud/gpt_5",
    "claude_opus_41": "https://jiale-de-deepseek.fc.chj.cloud/claude_opus_41",
    "gemini_2_5_pro": "https://linzhiyu-gemini.fc.chj.cloud/gemini_2_5_pro",
    "default": "http://10.80.12.172:8012/v1/chat/completions, http://10.80.12.172:8013/v1/chat/completions" # 新增您的模型
}

if __name__ == "__main__":
    # ================== 1. 配置区域 ==================
    model_name = 'default'
    input_file = "/lpai/volumes/base-ov-ali-sh-mix/zhaojiale/askQ/data/train_data/single_turn/sample_20k_gpt_oss_120b.jsonl"
    out_file = "/lpai/volumes/base-ov-ali-sh-mix/zhaojiale/askQ/data/gpt_res/sample_20k_gpt_oss_120b_2turn.jsonl"

    config = ZnyConfig(
        # URL会自动从 ZNY_API_URLS 字典中获取
        url=ZNY_API_URLS[model_name],
        model_name=model_name,
        temperature=0.7,
        max_tokens=16000,
        max_retries=10,
        retry_until_success=False, 
        qps=100, # 根据您的API服务器承受能力调整
        max_concurrent=100, # 根据您的API服务器承受能力调整
        asyncio_flag=True,
        image_flag=False,
        image_column_name=None,
        input_column_name='prompt',
        response_column_name='gpt_res_2', # 可以自定义输出列名
        resume_from_output=True,
        save_interval=50,
    )
    
    # 完整的 template 内容 (保持不变)
    # template = '''{query}'''
    template = '''You are an expert in generating conversational data. Your task is to create a two-turn dialogue based on an ambiguous initial question. The goal is to simulate a scenario where a helpful AI, instead of guessing, asks for clarification, and the user then provides the necessary information to get a correct answer.

**You must strictly follow these steps:**

1.  **Analyze the Input:**
    - You will be given an ambiguous/incomplete question (`degraded_question`).
    - You will also be given a detailed explanation of why it is ambiguous (`degraded_info`). This explanation tells you exactly what critical information was removed and what terms were made vague.
    - You will be given the correct final answer to the *original*, non-degraded question (`answer`).

2.  **Step 1: Generate the AI's Clarifying Question (the `ask` field).**
    - Act as a helpful but cautious AI assistant.
    - Read the `degraded_question` and use the `degraded_info` to identify the specific points of ambiguity or missing information.
    - Formulate a polite, natural-sounding question that asks the user to provide the exact information needed to resolve the ambiguity.
    - **Do not attempt to answer the question.** Your only goal is to seek clarification. For example, ask "Could you please specify what you mean by 'a certain pattern'?" or "To give you the most accurate answer, could you tell me the specific medical term you're referring to?".

3.  **Step 2: Generate the User's Follow-up and the AI's Final Answer (the `question_2` and `answer_2` fields).**
    - **a. Formulate the User's Clarifying Response (`question_2`):**
        - Now, switch roles and act as the user.
        - Your response should directly and concisely answer the AI's clarifying question from the previous step.
        - Use the `degraded_info` to find the *original, precise information* that was removed or obfuscated. This is what the user provides. For example: "Oh, sorry. I meant 'Onion-skin fibrosis'." or "Yes, the specific condition I'm asking about is Primary Biliary Cirrhosis."
    - **b. Formulate the AI's Final, Correct Answer (`answer_2`):**
        - Switch back to the role of the AI assistant.
        - Now that you have the complete, unambiguous information (from the `degraded_question` + `question_2`), provide the final, correct answer.
        - This answer **must** match the provided `answer` field.

4.  **Output Format:**
    - Return the result **only** in the following JSON structure.
    - **Do not include any explanations, comments, or markdown formatting outside of this JSON.**

    ```json
    {
        "ask": "<The AI's clarifying question from Step 2>",
        "question_2": "<The user's response providing the missing information from Step 3a>",
        "answer_2": "<The AI's final, correct answer from Step 3b>"
    }
    ```

**Here is the data to process:**

**Ambiguous Question**
{degraded_question}

**Degradation Info**
{degraded_info}

**Correct Final Answer**
{answer}
'''

    placeholder_mappings = {
        "text": "llm_prompt",
        "expected_answer": "answer",
        "query": "query",
        "degraded_question": "degraded_question",
        "degraded_info": "degraded_info",
        "answer": "answer",
    }
    
    # ================== 2. 执行流程 (保持不变) ==================
    print("--- 任务开始 ---")
    
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"已创建目录: {out_dir}")

    print("\n[步骤 1/4] 正在读取输入数据...")
    df = read_data(input_file)
    if df.empty:
        print("输入数据为空或无法读取，程序终止。")
        exit()
    
    print("\n[步骤 2/4] 正在根据模板填充Prompt...")
    prompt_df = fill_prompt_by_key_mappings(df, template, placeholder_mappings, prompt_key=config.input_column_name)
    print(f"已为 {len(prompt_df)} 条记录生成Prompt。")
    
    print("\n[步骤 3/4] 正在初始化并调用LLM API...")
    call_zny = CallLLMByZny(config)
    final_df = call_zny.get_gpt4api_df(prompt_df, out_file)
    
    print("\n[步骤 4/4] 正在进行最终校验和保存...")
    initial_count = len(prompt_df)
    final_count = len(final_df) if final_df is not None else 0

    if initial_count != final_count:
        print("\n" + "="*60)
        print("⚠️  警告: 数据量不匹配！ ⚠️")
        print(f"    - 原始输入数据量: {initial_count}")
        print(f"    - 最终输出数据量: {final_count}")
        print(f"    - 仍有 {initial_count - final_count} 条记录未成功处理。")
        print("    - 请检查日志中的错误信息，或重新运行此脚本以处理剩余的任务。")
        print("="*60 + "\n")
    else:
        print(f"\n✅ 校验通过：所有 {initial_count} 条记录均已成功处理并保存。\n")

    if final_df is not None and not final_df.empty:
        json_out_path = out_file.replace('.jsonl', '.json')
        try:
            final_df.to_json(json_out_path, indent=2, force_ascii=False, orient='records')
            print(f"已将最终结果转换为标准JSON格式: {json_out_path}")
        except Exception as e:
            print(f"转换为标准JSON时出错: {e}")
            
    print("--- 任务完成 ---")
