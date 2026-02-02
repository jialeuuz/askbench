# askQ：给 LLM 的仓库导航索引

这个文件的定位是：**给 LLM 快速定位“应该打开哪个子模块/哪个文档/哪个入口文件”**。
它会提供相对详细的“功能/入口/改哪里”的路线图，但不会展开到每个模块的具体实现细节（细节请读各模块的 `readme_for_ai`）。

英文版：`readme_for_ai.md`。

## 快速路由（先看这里）

| 用户任务是... | 先读 | 再读（LLM 细节导读） |
| --- | --- | --- |
| 跑评测（单轮 + AskBench 多轮） | `ask_eval/README_zh.md` | `ask_eval/readme_for_ai_zh.md` |
| 构建 AskBench 风格训练对话数据（离线） | `data_pipeline/README_zh.md` | `data_pipeline/readme_for_ai_zh.md` |
| RLVR reward / 接入 VERL | `reward/readme_zh` | `reward/readme_for_ai_zh.md` |
| 用 vLLM 部署 OpenAI-compatible API | `tools/vllm.sh` | 根目录 `readme_zh.md`（Tools 小节） |
| 训练 checkpoint 转换（给 vLLM 推理用） | `tools/merge.sh` | 根目录 `readme_zh.md`（Tools 小节） |
| 论文术语/背景 | `paper.pdf` | 根目录 `readme_zh.md` |

## 概念对齐（对应论文）

- **AskBench**：交互式 benchmark，把标准 QA 变成 judge-driven 的多轮协议（候选模型追问 → judge 只透露被问到的信息 → 候选模型作答 → judge 评分）。
- **AskMind**：意图缺失/信息不足维度；用 `required_points` checklist 来衡量“问得对不对/覆盖多少关键缺口”。
- **AskOverconfidence**：过度自信/错误前提维度；用 `misleading_points` checklist 来衡量“是否识别/纠正误导点并避免不当确定性”。
- **Rubric-guided RLVR**：turn-level shaping；奖励 checklist 覆盖，强惩罚“缺信息就直接给最终答案”，最终轮再奖励正确性。

## 常见工作流（高层）

### A）通过 API 评测本地模型

1）(可选) 把训练 checkpoint 转成可推理 HF 目录：`tools/merge.sh`  
2）用 vLLM 部署 OpenAI-compatible API：`tools/vllm.sh`  
3）在 `ask_eval/config/base.ini`（或 `ask_eval/run.sh`）里配置 API URL  
4）运行：`cd ask_eval && ./run.sh`（或 `python scripts/main.py --config config/base.ini`）

### B）离线构建 AskBench 风格训练对话数据

1）准备输入 JSONL（至少包含 `ori_question` + `expected_answer`，可选 `solution`）  
2）选择 `data_pipeline/strategies.py` 里的策略  
3）运行 `data_pipeline/main.py` 生成成功 JSONL + `_failed.jsonl`（用于断点续跑/排查 prompt）

### C）用 VERL + RLVR reward 做训练

1）把 reward 脚本拷到 VERL 并在 `default_compute_score()` 注册（见 `reward/readme_zh`）  
2）在 reward 文件里配置 judge 模型的 API（`API_URLS` / `JUDGE_MODEL_NAME`）  
3）`reward/train.sh` 是**已脱敏**的训练启动脚本参考（路径/数据都是占位符）

## 模块地图（打开什么 / 改哪里）

### `ask_eval/`（评测）

- 用途：跑单轮基准 + AskBench 风格多轮 judge loop。
- 主要入口：
  - `ask_eval/run.sh`（推荐：用变量覆盖 `config/base.ini`）
  - `ask_eval/scripts/main.py`（INI 驱动的总入口，按 task 分发）
- 关键配置：
  - `ask_eval/config/base.ini`（candidate/judge API；任务；并发；max_turns）
  - `ask_eval/config/common/*.ini`（任务级覆盖）
- 输出目录：`ask_eval/results/...`
- LLM 导读：`ask_eval/readme_for_ai_zh.md`

### `data_pipeline/`（数据构建）

- 用途：把 raw QA 转成 AskBench 风格多轮对话训练数据（或相关变体）。
- 主要入口：
  - `data_pipeline/main.py`（单 job）
  - `data_pipeline/main_queue.py` + `data_pipeline/run_queue.sh`（可选：多 job 调度）
- 关键点：
  - 策略：`data_pipeline/strategies.py`
  - prompt 模板：`data_pipeline/prompts.txt`
  - API 客户端：`data_pipeline/post_api.py`
- 输出：成功 JSONL + `_failed.jsonl`
- LLM 导读：`data_pipeline/readme_for_ai_zh.md`

### `reward/`（rubric-guided RLVR rewards）

- 用途：与论文对齐的两个维度 reward + 训练脚本参考（可接入 VERL）。
- 关键文件：
  - `reward/ask_mind_qa.py`（入口：`compute_score_ask_mind_qa`）
  - `reward/overconfidence_qa.py`（入口：`compute_score_overconfidence_qa`）
  - `reward/train.sh`（脱敏的训练启动参考）
- 接入说明：`reward/readme_zh`
- LLM 导读：`reward/readme_for_ai_zh.md`

### `tools/`（运维：转换 + 部署）

- `tools/merge.sh`：将训练产物（例如分片 checkpoint）合并/导出为可被 vLLM 加载的 HuggingFace 模型目录。
- `tools/vllm.sh`：启动 vLLM 的 OpenAI-compatible server，用 API 形式提供本地模型。

## 命名约定

- `README.md`：面向用户的“怎么跑”。
- `readme_for_ai.md`：面向 LLM 的实现导读（便于定位与改码）。
- `*_zh.*`：中文文档。
