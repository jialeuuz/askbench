# ask_eval 评测框架说明

readme记录着评测框架的架构信息，run.sh是评测的入口。

本仓库提供了一套针对多模任务的统一评测脚手架，既支持传统单轮基准（Math500、MedQA 等），也支持 AskBench 等由裁判模型驱动的多轮对话评测。本文档旨在快速梳理整体架构、关键模块与使用方式，方便后续查阅。

## 快速开始

```bash
# 1. 安装依赖（建议使用虚拟环境）
pip install -e .

# 2. 按需修改基础配置
vim config/base.ini

# 3. 启动评测（示例）
python scripts/main.py --config config/base.ini
# 或者使用 run.sh：先在 run.sh 顶部修改变量，再直接运行
./run.sh
```

运行结束后，结果会写入 `results/<task>/<task_name>/`，并追加汇总条目到 `results/final_result.txt`。