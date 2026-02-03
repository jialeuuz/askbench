# data_pipeline

从单轮 QA 样本构建 AskBench 风格的多轮对话数据（AskMind / AskOverconfidence），并提供可选的“直接回答→裁判→纠错”短流程策略。

除训练数据外，该 pipeline 也可用于把其他常规 QA bench 快速改造成 **AskMind/AskOverconfidence 风格的评测数据**（生成变体问题 + rubric/checklist），从而更容易拓展到新领域。

- **AskMind（缺失信息 / intent-deficient）**：把原始问题劣化为 `degraded_question`，同时生成缺失点清单 `required_points`，再进行多轮追问与用户模拟，最后作答并由裁判判断；必要时强制修正以保证最终答案正确。
- **AskOverconfidence（错误前提 / misleading claims）**：在保留原题 givens 的同时，注入“自信但错误”的断言，得到 `overconfidence_question` 与 `misleading_points`，再围绕误导点进行纠偏与作答。

更完整的实现细节（代码结构、Prompt 变量注入、策略内部步骤等）见 `readme_for_ai.md`。

## 环境准备

- Python 3.9+
- 安装依赖（任选其一）：
  - 在仓库根目录：`pip install -r data_pipeline/requirements.txt`
  - 或进入目录：`cd data_pipeline && pip install -r requirements.txt`
- 需要可用的 Chat Completions API（OpenAI 兼容返回：`choices[0].message.content`）

## 输入与输出

输入文件：JSONL（每行一个样本），至少包含：

```json
{
  "id": "可选；缺失时会按内容生成确定性哈希ID",
  "ori_question": "原始问题",
  "expected_answer": "标准答案",
  "solution": "可选；用于强制修正时的参考解题过程"
}
```

输出文件：JSONL（成功样本）。常见字段：

- `conversation_history`：多轮对话列表，元素形如 `{ "role": "user|assistant", "content": "..." }`
- `degraded_question` / `degraded_info` / `required_points`：AskMind（缺失信息）相关
- `overconfidence_question` / `overconfidence_info` / `misleading_points`：AskOverconfidence（错误前提）相关

失败文件：与输出同名追加 `_failed` 后缀，JSONL。每条含 `_failure` 元信息（失败步骤、原因、重试次数、截断的回复预览等）。

## 运行方式

最简单方式：直接修改 `data_pipeline/main.py` 末尾的常量，然后运行：

```bash
python data_pipeline/main.py
```

也可以在你自己的脚本中调用 `main()`（注意这是 async 函数）：

```python
import asyncio
import sys

# 让 Python 能找到 data_pipeline/ 下的脚本式模块（main.py / strategies.py / post_api.py ...）
sys.path.append("data_pipeline")
from main import main

asyncio.run(main(
    strategy="generate_multi_turn_degraded_training_data",
    input_file="/path/to/input.jsonl",
    output_file="/path/to/output.jsonl",
    prompts_file="data_pipeline/prompts.txt",
    api_urls=["http://host:port/v1/chat/completions"],
    api_type="default",
    api_token="none",
    max_concurrent_requests=200,
    timeout=3600,
    batch_size=1000,
    id_key="id",
    reprocess_failed=False,
))
```

断点续跑：默认会跳过已写入成功/失败文件的 `id`。如需重跑历史失败项，可设置环境变量 `REPROCESS_FAILED=1`。

## 策略（strategies）

在 `data_pipeline/strategies.py` 中实现，当前内置：

1) `generate_degraded_question_and_info`：生成 `degraded_question` / `degraded_info` / `required_points`
2) `generate_overconfidence_question_and_info`：生成 `overconfidence_question` / `overconfidence_info` / `misleading_points`
3) `generate_multi_turn_degraded_training_data`：AskMind 多轮对话生成（追问→用户模拟→覆盖检查→作答→裁判→必要时强制修正）
4) `generate_multi_turn_overconfidence_training_data`：AskOverconfidence 多轮对话生成（围绕误导点纠偏后再作答）
5) `strategy_direct_answer_and_correct`：短流程：直接回答→裁判；若错误则基于 `expected_answer`（可选结合 `solution`）重构“完美答案”

## 并行调度（可选）

`data_pipeline/run_queue.sh` + `data_pipeline/main_queue.py` 支持把多个任务队列以进程级并行的方式调度执行（适合多路 API 或多份数据并跑）。
