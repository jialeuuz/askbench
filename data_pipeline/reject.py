# -*- coding: utf-8 -*-
import json
import re
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging
import asyncio
import aiohttp
from asyncio import Semaphore
from tqdm import tqdm
import pickle
from dataclasses import dataclass, field

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================
# Checkpoint / resume
# =========================
class CheckpointManager:
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = self.load()

    def load(self) -> Dict:
        p = Path(self.checkpoint_file)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Loaded checkpoint: {self.checkpoint_file}")
                logger.info(f"Completed: {data.get('completed_count', 0)} / {data.get('total_count', 0)}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. A new checkpoint will be created.")
                return {}
        return {}

    def save(self, data: Dict):
        p = Path(self.checkpoint_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(data, f)

    def clear(self):
        p = Path(self.checkpoint_file)
        if p.exists():
            p.unlink()
            logger.info("Checkpoint file cleared")


# =========================
# Data structures
# =========================
@dataclass
class Sample:
    question: str
    expected_answer: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskItem:
    sample: Sample
    sample_index: int
    task_id: str


# =========================
# API caller
# =========================
class LLMCaller:
    def __init__(self, api_url: str, model: str = "default"):
        self.api_url = api_url
        self.model = model

    async def _post(self, session: aiohttp.ClientSession, payload: Dict, total_timeout=3600) -> Optional[Dict]:
        headers = {"Content-Type": "application/json"}
        try:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=total_timeout)
            ) as resp:
                if resp.status == 200:
                    try:
                        return await resp.json()
                    except Exception as e:
                        txt = await resp.text()
                        logger.error(f"JSON parse failed: {e}; first 500 chars: {txt[:500]}")
                        return None
                else:
                    try:
                        txt = await resp.text()
                        logger.error(f"API call failed status={resp.status}, first 500 chars of body={txt[:500]}")
                    except Exception as e:
                        logger.error(f"Failed to read error response: {e}")
                    return None
        except asyncio.TimeoutError as e:
            logger.error(f"API call timed out: {e}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unknown API error: {e}")
            return None

    async def chat(self, session: aiohttp.ClientSession, messages: List[Dict[str, str]],
                   temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        result = await self._post(session, payload)
        if result and "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"]
            if content:
                return content
        return None


# =========================
# Answer extraction (regex)
# =========================
def extract_answer_letter(text: str) -> Optional[str]:
    """
    Extract an answer letter (A/B/C/D, etc.) from text.

    Supported formats include:
    - "The answer is D"
    - "The answer is: D"
    - "Answer: D"
    - The last standalone uppercase letter in the text
    """
    if not text:
        return None
    
    # Prefer standard patterns
    patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*([A-Z])",  # The answer is D / Answer is: D
        r"(?:correct\s+)?(?:answer|choice|option)\s*:?\s*([A-Z])",  # Answer: D / Choice: D
        r"\b([A-Z])\s*(?:is\s+(?:the\s+)?(?:correct|right)\s+answer)",  # D is the correct answer
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Fallback: last standalone uppercase letter (often at the end)
    matches = re.findall(r'\b([A-Z])\b', text)
    if matches:
        return matches[-1]
    
    return None


def check_answer(llm_response: str, expected_answer: str) -> Dict[str, Any]:
    """
    Check correctness via regex-based answer extraction.
    """
    llm_letter = extract_answer_letter(llm_response)
    exp_letter = extract_answer_letter(expected_answer)
    
    if llm_letter is None:
        return {
            "llm_answer_letter": None,
            "expected_answer_letter": exp_letter,
            "is_correct": False,
            "error": "failed_to_extract_llm_answer"
        }
    
    if exp_letter is None:
        return {
            "llm_answer_letter": llm_letter,
            "expected_answer_letter": None,
            "is_correct": False,
            "error": "failed_to_extract_expected_answer"
        }
    
    is_correct = (llm_letter == exp_letter)
    
    return {
        "llm_answer_letter": llm_letter,
        "expected_answer_letter": exp_letter,
        "is_correct": is_correct
    }


# =========================
# Business logic (prompt template)
# =========================
def build_infer_prompt(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Answer the multiple-choice question. Use reasoning, then finish with the format: The answer is X"},
        {"role": "user", "content": f"Please answer the following multiple choice question. Provide your answer in the format \"The answer is X\" where X is the letter of your choice.\n\nQuestion:\n{question}\n\nPlease think step by step and provide your final answer."}
    ]


# =========================
# File writing (sync helper)
# =========================
def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# Main pipeline
# =========================
class Pipeline:
    def __init__(
        self,
        input_file: str,
        sample_times: int,
        success_output_file: str,
        failed_output_file: str,
        # Inference config
        inf_api_url: str,
        inf_model: str = "default",
        inf_max_concurrent: int = 100,
        # Other
        save_interval: int = 2000,
        use_checkpoint: bool = True,
        checkpoint_file: Optional[str] = None,
        max_retries_inf: int = 20,
        total_timeout_sec: int = 7200
    ):
        self.input_file = input_file
        self.sample_times = sample_times
        self.success_output_file = success_output_file
        self.failed_output_file = failed_output_file

        self.inf_api_url = inf_api_url
        self.inf_model = inf_model
        self.inf_max_concurrent = inf_max_concurrent

        self.save_interval = save_interval
        self.use_checkpoint = use_checkpoint
        self.checkpoint_file = checkpoint_file or (str(Path(success_output_file).with_suffix(".checkpoint")))
        self.max_retries_inf = max_retries_inf
        self.total_timeout_sec = total_timeout_sec

        # Stats
        self.total_tasks = 0
        self.success_count = 0
        self.failed_count = 0
        self.retry_count = 0
        self.inf_inflight = 0
        self.inf_done = 0

        self.stats_lock = asyncio.Lock()

        # Persistence buffers
        self.buffer_success: List[Dict] = []
        self.buffer_failed: List[Dict] = []
        self.persist_lock = asyncio.Lock()
        self.save_in_progress = False

        self.checkpoint_manager = CheckpointManager(self.checkpoint_file) if self.use_checkpoint else None

    def _load_questions(self) -> List[Sample]:
        path = Path(self.input_file)
        if not path.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return []
        out: List[Sample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    d = json.loads(s)
                    q = d.get("ori_question", "")
                    a = d.get("expected_answer", "")
                    meta = {k: v for k, v in d.items() if k not in ("ori_question", "expected_answer")}
                    if q and a:
                        out.append(Sample(q, a, meta))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON on line {line_num}")
        logger.info(f"Loaded {len(out)} questions")
        return out

    def _expand_tasks(self, samples: List[Sample]) -> List[TaskItem]:
        tasks: List[TaskItem] = []
        for qi, s in enumerate(samples):
            for si in range(self.sample_times):
                tid = f"q{qi}_s{si}"
                tasks.append(TaskItem(sample=s, sample_index=si + 1, task_id=tid))
        return tasks

    async def _save_buffers_and_checkpoint(self, current_completed: int):
        """Async save."""
        async with self.persist_lock:
            if self.buffer_success or self.buffer_failed:
                bufs = self.buffer_success[:]
                buff = self.buffer_failed[:]
                self.buffer_success.clear()
                self.buffer_failed.clear()
                await asyncio.to_thread(append_jsonl, self.success_output_file, bufs)
                await asyncio.to_thread(append_jsonl, self.failed_output_file, buff)

            if self.use_checkpoint and self.checkpoint_manager:
                ck = {
                    "completed_count": current_completed,
                    "total_count": self.total_tasks,
                    "success_count": self.success_count,
                    "failed_count": self.failed_count,
                    "timestamp": datetime.now().isoformat()
                }
                await asyncio.to_thread(self.checkpoint_manager.save, ck)

    async def run(self):
        samples = self._load_questions()
        all_tasks = self._expand_tasks(samples)
        self.total_tasks = len(all_tasks)
        if self.total_tasks == 0:
            logger.error("No tasks to process")
            return

        # Resume from checkpoint
        start_completed = 0
        if self.use_checkpoint and self.checkpoint_manager and self.checkpoint_manager.checkpoint_data:
            ck = self.checkpoint_manager.checkpoint_data
            start_completed = ck.get("completed_count", 0)
            self.success_count = ck.get("success_count", 0)
            self.failed_count = ck.get("failed_count", 0)

            if start_completed > 0:
                all_tasks = all_tasks[start_completed:]
                logger.info(f"Resuming from checkpoint: skipped {start_completed} tasks, remaining: {len(all_tasks)}")

        # Pre-clear output files (only for a fresh run)
        if start_completed == 0:
            Path(self.success_output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self.failed_output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.success_output_file, "w"):
                pass
            with open(self.failed_output_file, "w"):
                pass

        # aiohttp session
        connector = aiohttp.TCPConnector(
            limit=self.inf_max_concurrent + 100,
            limit_per_host=self.inf_max_concurrent + 50,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(
            total=self.total_timeout_sec,
            connect=60,
            sock_connect=60,
            sock_read=3600
        )

        # Semaphore
        inf_sem = Semaphore(self.inf_max_concurrent)

        # Caller
        inf_caller = LLMCaller(self.inf_api_url, self.inf_model)

        start_t = time.time()

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Progress bar
            pbar = tqdm(
                total=self.total_tasks,
                initial=start_completed,
                desc="Processing",
                dynamic_ncols=True
            )

            # Periodically update progress bar status
            async def tick_status():
                while True:
                    await asyncio.sleep(0.2)
                    elapsed = max(time.time() - start_t, 1e-6)
                    completed = self.success_count + self.failed_count
                    qps = completed / elapsed
                    async with self.stats_lock:
                        postfix = (
                            f"Inf={self.inf_inflight}, "
                            f"Done={self.inf_done}, "
                            f"Success={self.success_count}, "
                            f"Failed={self.failed_count}, "
                            f"QPS={qps:.2f}"
                        )
                    pbar.set_postfix_str(postfix)

            status_task = asyncio.create_task(tick_status())

            # Worker: inference + regex check
            async def inf_worker(task: TaskItem):
                retries = 0
                content = None
                error_msg = None

                # Inference phase (with retries)
                while retries <= self.max_retries_inf:
                    async with inf_sem:
                        async with self.stats_lock:
                            self.inf_inflight += 1
                        try:
                            content = await inf_caller.chat(
                                session,
                                build_infer_prompt(task.sample.question),
                                temperature=0.7,
                                max_tokens=16000
                            )
                        finally:
                            async with self.stats_lock:
                                self.inf_inflight -= 1

                    if content is not None:
                        break
                    retries += 1
                    error_msg = "infer_failed"
                    async with self.stats_lock:
                        self.retry_count += 1
                    await asyncio.sleep(min(2 * retries, 5))

                async with self.stats_lock:
                    self.inf_done += 1

                # Inference failed
                if content is None:
                    sample = task.sample
                    row = {
                        **sample.meta,
                        "ori_question": sample.question,
                        "expected_answer": sample.expected_answer,
                        "llm_response": None,
                        "error": error_msg or "infer_failed",
                        "sample_index": task.sample_index,
                        "task_id": task.task_id,
                        "timestamp": datetime.now().isoformat(),
                        "stage": "infer"
                    }
                    self.buffer_failed.append(row)
                    async with self.stats_lock:
                        self.failed_count += 1
                    pbar.update(1)
                    await maybe_save()
                    return

                # Regex-based check (no extra API calls)
                check_result = check_answer(content, task.sample.expected_answer)
                is_correct = check_result.get("is_correct", False)

                sample = task.sample
                final_row = {
                    **sample.meta,
                    "ori_question": sample.question,
                    "expected_answer": sample.expected_answer,
                    "llm_response": content,
                    "check_result": check_result,
                    "sample_index": task.sample_index,
                    "task_id": task.task_id,
                    "timestamp": datetime.now().isoformat()
                }

                if is_correct:
                    self.buffer_success.append(final_row)
                    async with self.stats_lock:
                        self.success_count += 1
                else:
                    self.buffer_failed.append(final_row)
                    async with self.stats_lock:
                        self.failed_count += 1

                pbar.update(1)
                await maybe_save()

            # Debounced persistence
            async def maybe_save():
                completed = self.success_count + self.failed_count
                if (completed % self.save_interval == 0) or (completed == self.total_tasks):
                    if not self.save_in_progress:
                        self.save_in_progress = True
                        try:
                            await self._save_buffers_and_checkpoint(completed)
                        finally:
                            self.save_in_progress = False

            # Create all tasks at once
            logger.info(f"Creating {len(all_tasks)} inference tasks...")
            inf_tasks = [asyncio.create_task(inf_worker(t)) for t in all_tasks]
            logger.info("All inference tasks submitted; running concurrently...")

            # Wait for completion
            await asyncio.gather(*inf_tasks)

            # Stop status task
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass

            # Final flush
            await self._save_buffers_and_checkpoint(self.success_count + self.failed_count)
            if self.use_checkpoint and (self.success_count + self.failed_count) == self.total_tasks:
                self.checkpoint_manager.clear()
            pbar.close()

        logger.info("Sampling finished!")
        logger.info(f"Successful samples: {self.success_count}")
        logger.info(f"Failed samples: {self.failed_count}")
        logger.info(f"Successful samples saved to: {self.success_output_file}")
        logger.info(f"Failed samples saved to: {self.failed_output_file}")


# =========================
# Sync entrypoint
# =========================
def run_pipeline(
    input_file: str,
    sample_times: int,
    success_output_file: str,
    failed_output_file: str,
    inf_api_url: str,
    inf_model: str = "default",
    inf_max_concurrent: int = 100,
    save_interval: int = 2000,
    use_checkpoint: bool = True,
    checkpoint_file: Optional[str] = None,
    max_retries_inf: int = 20,
    total_timeout_sec: int = 7200
):
    pipeline = Pipeline(
        input_file=input_file,
        sample_times=sample_times,
        success_output_file=success_output_file,
        failed_output_file=failed_output_file,
        inf_api_url=inf_api_url,
        inf_model=inf_model,
        inf_max_concurrent=inf_max_concurrent,
        save_interval=save_interval,
        use_checkpoint=use_checkpoint,
        checkpoint_file=checkpoint_file,
        max_retries_inf=max_retries_inf,
        total_timeout_sec=total_timeout_sec
    )
    asyncio.run(pipeline.run())


# =========================
# Script entrypoint
# =========================
if __name__ == "__main__":
    INPUT_JSONL_FILE = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train-medmcqa-clear-sample2w.jsonl"
    SAMPLE_TIMES = 16
    SUCCESS_OUTPUT_FILE = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/yitu/medmcqa_all_reject_235b_x16.jsonl"
    FAILED_OUTPUT_FILE = "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/yitu/medmcqa_all_reject_235b_x16_failed.jsonl"

    INF_API_URL = "http://10.80.128.219:9012/v1/chat/completions"

    run_pipeline(
        input_file=INPUT_JSONL_FILE,
        sample_times=SAMPLE_TIMES,
        success_output_file=SUCCESS_OUTPUT_FILE,
        failed_output_file=FAILED_OUTPUT_FILE,
        inf_api_url=INF_API_URL,
        inf_model="default",
        inf_max_concurrent=2000,     # uses a single endpoint; only one concurrency value needed
        save_interval=10000,
        use_checkpoint=True,
        checkpoint_file=None,
        max_retries_inf=20,
        total_timeout_sec=7200
    )
