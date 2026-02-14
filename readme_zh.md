<div align="center">

# When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification

[![论文](https://img.shields.io/badge/Paper-PDF-blue?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2602.11199v1)
[![arXiv](https://img.shields.io/badge/arXiv-2602.11199-b31b1b.svg)](https://arxiv.org/abs/2602.11199)
[![HuggingFace (Bench)](https://img.shields.io/badge/HuggingFace-askbench__bench-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/jialeuuz/askbench_bench)
[![HuggingFace (Train)](https://img.shields.io/badge/HuggingFace-askbench__train-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/jialeuuz/askbench_train)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[中文](readme_zh.md) | [English](README.md) | [LLM 导读](readme_for_ai_zh.md)

</div>

本仓库包含论文 **“When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification”** 的代码与相关资源。论文已发布在 arXiv：🔗 [abs](https://arxiv.org/abs/2602.11199) | [pdf](https://arxiv.org/pdf/2602.11199v1)。

大语言模型在面对**信息不足**或**包含误导前提**的提问时，往往仍会直接作答，从而产生幻觉或强化错误认知。本项目研究模型应该**何时**以及**问什么**来进行澄清，并提供：

- **AskBench**：一个交互式基准，将标准 QA 样本转换为带显式检查点的多轮交互。
- 一个**统一的 Judge Loop**：在评测中同时完成 (1) 最终答案评估，以及 (2) 当被测模型发起追问时模拟用户回复。
- 两个核心设置：
  - **AskMind**：意图缺失/信息不足的问题，需要通过追问获取关键信息后再回答。
  - **AskOverconfidence**：问题包含错误前提/误导断言，需要识别并纠正后再回答。

如果你希望借助 LLM 快速理解/修改代码结构（便于调试与定位入口），可先阅读 `readme_for_ai.md`（中文版：`readme_for_ai_zh.md`）。

## 📌 目录

- 🚀 评测： [运行评测](#evaluation)
- 🎯 训练： [RLVR reward + VERL 接入](#training)
- 🧪 data pipeline： [构建 AskBench 风格数据](#data-pipeline)
- 🛠️ 工具： [checkpoint 合并 + OpenAI-compatible 部署](#tools)
- 📦 数据集： [Hugging Face 链接](#datasets)

## ✨ AskBench 速览

AskBench 将“澄清”作为一种**交互能力**来评测。每个样本运行时包含：

- **被测模型**（assistant），以及
- **Judge 模型**（在多轮评测中承担多个角色）：
  - **模拟用户**（当 assistant 追问时补充信息），以及
  - **评分器**（判断最终答案是否正确、检查点是否覆盖充分）。

整体流程是：被测模型可提澄清问题 → Judge 视情况模拟用户回复 → 产出最终答案 → Judge 给出判定与统计。

## 🔎 为什么是 AskBench？

在真实交互中，用户问题常常 **信息不足** 或包含 **误导前提**。传统单轮 benchmark 更擅长衡量“答得对不对”，但很难衡量：

- 模型是否能在合适时机选择追问；以及
- 追问是否命中真正关键的缺失点/误导点。

AskBench 的设计旨在让“澄清能力”可规模化评测：

- **交互式且可自动化**：judge loop 在模型明确追问时才模拟用户补充信息，并端到端评分最终答案。
- **细粒度且可解释**：checkpoints/rubrics 把澄清行为拆成可分析的条目指标（例如 checkpoint coverage）。
- **高拓展性**：为标准 QA 生成“变体问题”（degraded 或注入误导前提）并配套 checklist，即可快速改造为交互式评测。
- **易用性强**：评测只依赖 OpenAI-compatible API（被测模型 + judge），可通过 vLLM 等工具本地部署。

## 📈 论文结果

论文中，rubric-guided RLVR 在 AskBench 多轮评测上显著提升澄清能力，同时能保持（甚至提升）单轮 QA 等通用能力。

### AskBench 多轮澄清（普通协议，Table 4）

指标说明：

- `acc`（accuracy）：最终答案是否正确（由 judge 评分）。
- `cov`（checkpoint coverage）：作答前对 checklist 的覆盖程度（AskMind 的 `required_points`；AskOverconfidence 的 `misleading_points`）。

| 模型 | AskMind `acc` | AskMind `cov` | AskOverconfidence `acc` | AskOverconfidence `cov` |
| --- | :---: | :---: | :---: | :---: |
| Gemini-2.5-Pro | 0.567 | 0.124 | 0.840 | 0.749 |
| GPT-4.1 | 0.495 | 0.118 | 0.730 | 0.602 |
| Qwen2.5-7B-Instruct | 0.332 | 0.214 | 0.443 | 0.188 |
| OursI | 0.615 | 0.679 | 0.628 | 0.641 |
| OursO | 0.617 | 0.807 | 0.548 | 0.894 |

### 严格两轮协议（“Hard”, Table 5）

严格两轮协议要求：第 1 轮必须澄清/纠偏，第 2 轮必须直接给最终答案（不能再追问）。

| 模型 | AskMind `acc` | AskMind `cov` | AskOverconfidence `acc` | AskOverconfidence `cov` |
| --- | :---: | :---: | :---: | :---: |
| Gemini-2.5-Pro | 0.0551 | 0.2206 | 0.0100 | 0.7350 |
| GPT-4.1 | 0.0352 | 0.2035 | 0.0000 | 0.5865 |
| Qwen2.5-7B-Instruct | 0.0176 | 0.1288 | 0.0050 | 0.1955 |
| OursI | 0.2714 | 0.5013 | 0.1975 | 0.5065 |
| OursO | 0.1965 | 0.4235 | 0.2600 | 0.7778 |

注：论文中 Gemini 指 Gemini-2.5-Pro，GPT 指 GPT-4.1，Qwen 指 Qwen2.5-7B-Instruct。
OursI/OursO 分别是我们在 AskMind/AskOverconfidence 维度上训练得到的模型。

单轮准确率与 HealthBench 得分（Table 3）：

| 模型 | Math500 | MedQA | HealthBench | GPQA-d | BBH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini-2.5-Pro | 0.952 | 0.943 | 0.649 | 0.864 | 0.946 |
| GPT-4.1 | 0.936 | 0.918 | 0.645 | 0.701 | 0.708 |
| Qwen2.5-7B-Instruct | 0.760 | 0.653 | 0.526 | 0.309 | 0.506 |
| OursI | 0.780 | 0.936 | 0.606 | 0.497 | 0.758 |
| OursO | 0.720 | 0.992 | 0.559 | 0.781 | 0.760 |

说明：部分评测（例如 HealthBench）属于 LLM-judge 评测。为节省成本并便于复现，我们使用开源 judge（如论文中使用的 Qwen3-30B-A3B-Instruct-2507）而非闭源 GPT judge，因此绝对分数可能与官方报告存在差异，但不同模型的整体排名趋势通常保持一致。

## 🧩 仓库结构

- `ask_eval/`：评测 pipeline（单轮 + AskBench 风格的多轮评测）。
  - 使用说明：`ask_eval/README.md`
  - 实现细节/调试定位：`ask_eval/readme_for_ai.md`
  - 入口脚本：`ask_eval/run.sh`
- `data_pipeline/`：数据构建 pipeline，用于生成 AskBench 风格的多轮对话数据（**训练 + 评测**），以及把常规 QA bench 快速改造为 AskMind/AskOverconfidence 风格的变体与 rubric/checklist。
  - 使用说明：`data_pipeline/README.md`
  - 实现细节/调试定位：`data_pipeline/readme_for_ai.md`
  - 入口脚本：`data_pipeline/main.py`
- `reward/`：rubric-guided reward / 训练辅助脚本（用于 RLVR 风格训练）。
- `tools/`：辅助脚本，用于（1）将训练 checkpoint 转成可推理的 HuggingFace 模型目录，以及（2）用 vLLM 部署 OpenAI-compatible API。
- `readme_for_ai.md`：面向 LLM 的仓库导读（架构梳理 + 关键入口）。
- `paper.pdf`：论文 PDF（匿名投稿版本构建产物；以 arXiv 版本为准）。

原中文文档已用 `_zh` 后缀保留（例如 `ask_eval/README_zh.md`）。

## ⚙️ 环境与安装

建议：Python 3.10+，并使用 conda 环境。

### 安装 `ask_eval`

```bash
conda create -n askq python=3.10 -y
conda activate askq

pip install -e ./ask_eval
```

### 安装 `data_pipeline` 依赖

```bash
pip install -r data_pipeline/requirements.txt
```

<a id="evaluation"></a>
## 🚀 Quickstart：运行评测（AskBench + 标准 QA）

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

<a id="tools"></a>
## 🛠️ 工具：checkpoint 转换 + OpenAI-compatible API 部署

`ask_eval` 通过 OpenAI-compatible 的 chat-completions API 调用模型。如果你的工作流是基于 API 调用，这里提供了 `tools/` 下两个常用脚本，对应一个常见流程：

1) （可选）**把训练 checkpoint 转成推理可用的 HuggingFace 模型目录**：`tools/merge.sh`。
2) **用 vLLM 部署成 OpenAI-compatible API**：`tools/vllm.sh`。

### 训练 checkpoint 转换（`tools/merge.sh`）

部分训练产物（例如 VERL/RLVR 训练输出的分片 checkpoint）无法直接被 vLLM 加载推理，需要先合并/导出成标准 HuggingFace 模型文件夹。

1) 修改 `tools/merge.sh` 中的变量：
   - `CHECKPOINT_DIR`：训练 checkpoint 路径（通常是某个 `.../actor` 目录）
   - `OUTPUT_DIR`：导出后的模型目录
   - `WORLD_SIZE`：checkpoint 分片数量（一般等于训练的 world size）
   - `MERGE_SCRIPT_PATH`：你环境中 `merge_verl.py` 转换脚本的路径
2) 运行：

```bash
bash tools/merge.sh
```

成功后，将 `tools/vllm.sh` 的 `MODEL_PATH` 指向导出的 `OUTPUT_DIR`。

### 用 vLLM 部署 OpenAI-compatible API（`tools/vllm.sh`）

该脚本启动 vLLM 的 OpenAI-compatible server（`vllm.entrypoints.openai.api_server`）。

1) 修改 `tools/vllm.sh` 中的变量：
   - `MODEL_PATH`：HuggingFace 模型目录（可以是 base 模型，也可以是 `tools/merge.sh` 产出的 `OUTPUT_DIR`）
   - `CUDA_DEVICES` 与 `TP`：应与参与 tensor-parallel 的 GPU 数量一致
   - `PORT`：服务端口
2) 运行：

```bash
bash tools/vllm.sh
```

然后在 `ask_eval/config/base.ini`（或 `ask_eval/run.sh`）中配置服务地址，例如：

- `[model] api_url = http://<host>:<port>/v1`
- `[model] model_name = default`（需与 `tools/vllm.sh` 中的 `--served-model-name` 一致）

<a id="datasets"></a>
## 📦 数据集

- **Hugging Face（推荐下载链接）**：
  - 🤗 AskBench 评测数据：[jialeuuz/askbench_bench](https://huggingface.co/datasets/jialeuuz/askbench_bench)
  - 🤗 AskMind/AskOverconfidence 训练轨迹：[jialeuuz/askbench_train](https://huggingface.co/datasets/jialeuuz/askbench_train)
- **评测数据（仓库跟踪）**：位于 `ask_eval/data/`（AskBench 子集 + pipeline 使用的常规 benchmark）。
- **可选训练/中间数据（不跟踪）**：可放在根目录的 `data/` 下（本仓库默认 `.gitignore` 忽略 `data/`）。

## 📝 输出（会写哪些文件）

根据任务类型，`ask_eval` 会写入以下文件的组合：

- `results.txt`：人类可读的汇总（指标 + 耗时）。
- `summary_results.json`：单轮任务的逐样本输出。
- `askbench_detailed_results.json`：AskBench 风格任务的逐轮对话轨迹与 judge 判定细节。

## 🧱 生成（或重建）AskBench 合并评测集

AskBench 的主任务通常是由多个子集拼成的小规模 mixture（例如每个来源 benchmark 采样 100 条）。

```bash
python ask_eval/data/ask_bench/ask_mind/build_combined_eval.py
python ask_eval/data/ask_bench/ask_overconfidence/build_combined_eval.py
```

<a id="data-pipeline"></a>
## 🧪 Quickstart：构建 AskBench 风格数据（训练 + 评测）

数据构建 pipeline 可以生成 AskBench 风格的多轮对话（澄清 → 模拟用户回复 → 作答 → 评审）用于训练；同时也可以把其他 QA bench 快速改造为 AskMind/AskOverconfidence 风格的评测数据（生成变体问题 + checklist/rubrics）。

具体入口与参数说明见 `data_pipeline/README.md`。

<a id="training"></a>
## 🎯 Rubric-guided reward（RLVR）

`reward/` 目录包含两个 **VERL 可直接使用** 的 reward 函数实现，对应论文中的 rubric-guided、turn-level shaping：

- AskMind（意图缺失 / 信息不足）：`reward/ask_mind_qa.py`（`data_source = ask_mind_qa`）
- AskOverconfidence（过度自信 / 错误前提）：`reward/overconfidence_qa.py`（`data_source = overconfidence_qa`）

使用方式是将脚本拷贝到 VERL 的 `verl/utils/reward_score/` 并在 `default_compute_score()` 里注册；judge 端点通过 `API_URLS` / `JUDGE_MODEL_NAME` 配置。更详细的接入步骤见 `reward/readme`，代码细节说明见 `reward/readme_for_ai_zh.md`。

另外提供了已脱敏的训练启动脚本参考（VERL + Ray + DAPO/GRPO）：`reward/train.sh`。

## 📚 引用

如果你使用了本仓库，请引用论文：

```bibtex
@misc{askbench2026,
  title        = {When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification},
  author       = {Anonymous},
  year         = {2026},
  note         = {Anonymous ACL submission},
}
```
