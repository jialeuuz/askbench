askQ: API-Based Language Model Evaluation Framework

askQ 是一个基于 API 调用的语言模型（LLM）评测框架。其核心设计是将模型的**生成（Generation）**与**评审（Judging）**过程解耦，通过独立的 API 进行交互。这使得框架非常灵活，可以轻松评测各类模型，并支持自定义的评审标准，尤其擅长处理需要多轮交互的复杂任务。
✨ 核心特性 (Core Features)

- API 驱动: 整个评测流程基于 API 调用，易于集成和扩展。
- 生成/评审分离: 将模型输出的生成过程和质量的评审过程解耦，允许使用不同的模型（如 GPT-4）作为“裁判”。
- 支持多轮交互: 内置模拟用户行为的“类人模型”，能够对模型进行多轮追问，评估其在连续对话中的表现。
- 灵活的 API 支持:
  - 支持标准的 **OpenAI API 格式**。
  - 支持**自定义内部 API 格式**，方便在企业内网环境中使用。
- 两种评测模式:
  - 单次评测: 针对单个模型进行快速、手动的评测。
  - 全自动评测: 通过配置文件，实现多模型、多任务的批量自动化评测。
📊 评测原理 (Evaluation Principle)

askQ 的工作流程旨在模拟真实的用户交互场景，通过“生成-评审-追问”的闭环来深度评估模型能力。

高层工作流

1. 生成模型 (Generator): 待评测的模型，负责根据输入（Prompt）生成回复。
2. 评审模型 (Judge): 负责评估生成内容是否满足任务要求，并决定评测是否结束。
3. 类人模型 (Human-like Model): 在多轮评测中，模拟用户根据当前对话上下文发起追问，驱动评测继续进行。
详细流程图

评测的核心流程，尤其是在多轮对话场景下，如下图所示：

<!--
    重要提示:
    请将下面的图片 workflow.png 上传到你的 GitHub 仓库中 (例如，在根目录创建一个 docs/images 文件夹),
    然后将下面的 src 路径替换为实际的图片链接。
-->
<p align="center">
  <img src="https://i.imgur.com/714155g.png" alt="askQ Workflow" width="700"/>
</p>

该流程可以分解为以下步骤：

4. 任务启动 (Initiation):
  - askBench 数据集提供初始 query 给待评测的 LLM。
  - 同时，将原始问题、上下文信息和标准答案等提供给“类人模型”和“评审模型”作为参考。
5. 模型生成 (Generation):
  - LLM 接收 query 并生成 LLM回复。
6. 评审与决策 (Judgment & Decision):
  - judge 模型 评估 LLM回复 的质量，并判断任务是否已达到“最终结果”状态。
  - 是最终结果: 评测结束，结果被记录到 最终结果统计 模块。
  - 不是最终结果: judge 模型 将判定结果和当前 上下文 传递给“类人模型”。
7. 多轮交互循环 (Multi-turn Interaction Loop):
  - “类人模型”根据收到的上下文，生成一个 新一轮query（追问）。
  - 这个新的 query 被发送给 LLM，开始新一轮的生成与评审，直到任务完成。
🚀 快速开始 (Quick Start)

1. 环境配置

我们建议使用 Conda 来管理 Python 环境。

# 1. 创建并激活 conda 环境
conda create -n eval python=3.10
conda activate eval

# 2. 安装 askQ 及其核心依赖
# 以可编辑模式安装，方便开发和调试
pip install -e .

2. 使用方法

askQ 支持两种评测模式，请根据你的需求选择。


---

模式一：单 API 评测 (手动)

此模式适合对单个模型进行快速验证和评测。

步骤 1: 部署待评测模型的 API 服务

首先，将待评测的模型部署为一个 API 服务。我们提供了基于 sglang 或 vllm 的部署脚本示例。

- 打开部署脚本，例如 models/sglang_r1.sh。
- 修改 MODEL_PATH 变量，指向你的模型文件路径。
# 示例: models/sglang_r1.sh
MODEL_PATH=/path/to/your/model
- 运行脚本以启动 API 服务。请记下服务的 IP 地址和端口。
步骤 2: 配置评测脚本

- 打开评测运行脚本 scripts/run.sh。
- 根据你的环境修改以下环境变量：
# 你的项目根目录
export PYTHONPATH=/path/to/your/askQ/project

# 步骤 1 中部署的待评测模型的 API 地址
export API_URL="http://<IP_ADDRESS>:<PORT>/v1/chat/completions"

# 要评测的任务名称，多个任务用逗号分隔 (例如: "ceval,mmlu")
export TASKS="your_task_name"

# 评测结果的保存目录
export SAVE_DIR="./outputs/results"

# 用于评审的 Judge 模型 API 地址 (如果任务需要)
export EVAL_API_URL="http://<JUDGE_IP>:<JUDGE_PORT>/v1/chat/completions"
步骤 3: 运行评测

建议在 tmux 或 screen 中运行，以防终端断开。

bash scripts/run.sh

评测完成后，结果将保存在你指定的 SAVE_DIR 目录下。


---

模式二：全自动评测 (批量)

此模式适合需要对多个模型、多个任务进行大规模批量评测的场景。

前提条件:
- 全自动评测需要在**部署 API 的服务器上**配置环境并运行。
- 需要安装额外的依赖包。
步骤 1: 安装额外依赖

# 建议将以下包装入一个 requirements.txt 文件后通过 pip 安装
pip install "sglang[all]" pybase64 pydantic orjson uvicorn uvloop fastapi zmq Pillow openai partial_json_parser transformers sentencepiece

步骤 2: 配置全自动评测脚本

- 打开主运行脚本 scripts/auto_run.sh。
- 主要配置以下参数：
# 描述模型路径和评测任务对应关系的 JSON 文件
MODEL_TASK_JSON="config/model_tasks.json"

# 所有评测结果的根目录
BASE_SAVE_DIR="./outputs/auto_results"

# askQ 项目的根目录
EVAL_PROJECT_ROOT="/path/to/your/askQ/project"
- 核心配置: 在 MODEL_TASK_JSON 文件中，你需要定义每个模型的路径以及它需要执行的评测任务列表。
步骤 3: 运行全自动评测

bash scripts/auto_run.sh

脚本将自动根据配置文件，依次加载模型、部署 API、执行评测并保存结果。

📝 关于 API 格式的说明

- OpenAI 格式: 当使用符合 OpenAI 标准的 API（无论是 OpenAI 官方、开源模型服务如 vLLM/SGLang，还是闭源模型）作为 Judge 模型时，功能完整。
- 内部 API 格式: 如果你使用公司内部的专有 API 格式，请注意：
  - 默认情况下，可能无法直接调用外部闭源模型（如 GPT-5）作为 Judge。
  - 你可以通过将 Judge 模型切换为一个**同样部署为 OpenAI 格式 API 的开源模型**来解决此问题。
