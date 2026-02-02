# When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification

[中文](readme_zh.md) | [English](readme)

本仓库包含论文 **“When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification”** 的代码与相关资源（见 `paper.pdf`）。

大语言模型在面对**信息不足**或**包含误导前提**的提问时，往往仍会直接作答，从而产生幻觉或强化错误认知。本项目研究模型应该**何时**以及**问什么**来进行澄清，并提供：

- **AskBench**：一个交互式基准，将标准 QA 样本转换为带显式检查点的多轮交互。
- 一个**统一的 Judge Loop**：在评测中同时完成 (1) 最终答案评估，以及 (2) 当被测模型发起追问时模拟用户回复。
- 两个核心设置：
  - **AskMind**：意图缺失/信息不足的问题，需要通过追问获取关键信息后再回答。
  - **AskOverconfidence**：问题包含错误前提/误导断言，需要识别并纠正后再回答。

## AskBench 速览

AskBench 将“澄清”作为一种**交互能力**来评测。每个样本运行时包含：

- **被测模型**（assistant），以及
- **Judge 模型**（在多轮评测中承担多个角色）：
  - **模拟用户**（当 assistant 追问时补充信息），以及
  - **评分器**（判断最终答案是否正确、检查点是否覆盖充分）。

整体流程是：被测模型可提澄清问题 → Judge 视情况模拟用户回复 → 产出最终答案 → Judge 给出判定与统计。

## 仓库结构

- `ask_eval/`：评测 pipeline（单轮 + AskBench 风格的多轮评测）。
  - 使用说明：`ask_eval/README.md`
  - 实现细节/调试定位：`ask_eval/readme_for_ai.md`
  - 入口脚本：`ask_eval/run.sh`
- `data_pipeline/`：数据构建 pipeline，用于生成 AskBench 风格的多轮对话训练数据。
  - 使用说明：`data_pipeline/README.md`
  - 实现细节/调试定位：`data_pipeline/readme_for_ai.md`
  - 入口脚本：`data_pipeline/main.py`
- `reward/`：rubric-guided reward / 训练辅助脚本（用于 RLVR 风格训练）。
- `paper.pdf`：论文 PDF（匿名投稿版本构建产物）。

原中文文档已用 `_zh` 后缀保留（例如 `ask_eval/README_zh.md`）。

## 环境与安装

建议：Python 3.10+，并使用虚拟环境。

### 安装 `ask_eval`

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ./ask_eval
```

### 安装 `data_pipeline` 依赖

```bash
pip install -r data_pipeline/requirements.txt
```

## Quickstart：运行评测（AskBench + 标准 QA）

`ask_eval` 假设你有一个 **OpenAI-compatible** 的 chat-completions API，分别用于：

- **被测模型**（candidate），以及
- **Judge 模型**（负责评分；在 AskBench 中还负责模拟用户）。

1) 在 `ask_eval/config/base.ini`（以及可选的 `ask_eval/config/common/` 任务级覆盖）中配置模型 endpoint 与 token。
2) 运行：

```bash
cd ask_eval
python scripts/main.py --config config/base.ini
```

如果希望通过 shell 变量覆盖配置项，可使用 `ask_eval/run.sh`。

说明：

- AskBench 风格任务通过 `ask_eval/scripts/run_ask.py` 跑 judge-driven 多轮评测。
- 可在 `ask_eval/run.sh` 中设置 `STRICT_MODE=1` 来启用更严格的两轮协议（第一轮必须澄清/纠正，第二轮必须直接给最终答案且不能再追问）。
- 评测输出写入 `ask_eval/results/<task>/<task_name>/`，并在 `ask_eval/results/final_result.txt` 追加聚合汇总行。

## 数据集

- **评测数据（仓库跟踪）**：位于 `ask_eval/data/`（AskBench 子集 + pipeline 使用的常规 benchmark）。
- **可选训练/中间数据（不跟踪）**：可放在根目录的 `data/` 下（本仓库默认 `.gitignore` 忽略 `data/`）。

## 输出（会写哪些文件）

根据任务类型，`ask_eval` 会写入以下文件的组合：

- `results.txt`：人类可读的汇总（指标 + 耗时）。
- `summary_results.json`：单轮任务的逐样本输出。
- `askbench_detailed_results.json`：AskBench 风格任务的逐轮对话轨迹与 judge 判定细节。

## 生成（或重建）AskBench 合并评测集

AskBench 的主任务通常是由多个子集拼成的小规模 mixture（例如每个来源 benchmark 采样 100 条）。

```bash
python ask_eval/data/ask_bench/ask_mind/build_combined_eval.py
python ask_eval/data/ask_bench/ask_overconfidence/build_combined_eval.py
```

## Quickstart：构建 AskBench 风格训练对话数据

数据构建 pipeline 会生成多轮对话（澄清 → 模拟用户回复 → 作答 → 评审），并把成功样本与失败元信息一起写出，便于断点续跑与排查问题。

具体入口与参数说明见 `data_pipeline/README.md`。

## Rubric-guided reward（RLVR）

`reward/` 目录包含一个 reward 实现与若干辅助脚本，用于 rubric-guided 的 RLVR 风格训练。详见 `reward/readme`。

## 引用

如果你使用了本仓库，请引用论文：

```bibtex
@misc{askbench2026,
  title        = {When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification},
  author       = {Anonymous},
  year         = {2026},
  note         = {Anonymous ACL submission},
}
```
