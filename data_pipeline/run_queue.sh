#!/bin/bash

# ==============================================================================
# --- 任务配置中心 (可读性优化版) ---
# ==============================================================================

# 使用方法:
# 1. 为每一个【串行任务队列】定义一个变量 (例如 SERIAL_QUEUE_1, SERIAL_QUEUE_2)。
#    - 每个变量包含一个或多个任务的JSON数组 `[{...}, {...}]`。
#    - 队列内的任务会按顺序执行。

# 2. 在 `PARALLEL_QUEUES` 列表中组合你想要【并行执行】的队列。
#    - 列表中的每个队列变量都会在一个独立的并行进程中运行。

# --- 串行队列 1 ---
SERIAL_QUEUE_1='[
  {
    "STRATEGY": "generate_multi_turn_training_data",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/dapo/dapo_distill_sample_2k.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_math_sample_20k_oss120b_heigh.jsonl",
    "API_URLS": ["http://10.80.13.242:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "generate_multi_turn_training_data",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/med_sample_20k_clear.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_med_sample_20k_oss120b_heigh.jsonl",
    "API_URLS": ["http://10.80.13.242:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "strategy_direct_answer_and_correct",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/med_sample_20k_clear.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/ori_med_sample_20k_oss120b_heigh.jsonl",
    "API_URLS": ["http://10.80.13.242:8012/v1/chat/completions"]
  }
]'

# --- 串行队列 2(使用不同API，可以并行) ---
SERIAL_QUEUE_2='[
  {
    "STRATEGY": "generate_multi_turn_training_data",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/useless/math_sample_20k.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_math_sample_20k_a3b_ins_2507.jsonl",
    "API_URLS": ["http://10.80.12.34:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "strategy_direct_answer_and_correct",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/useless/math_sample_20k.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/ori_math_sample_20k_a3b_ins_2507.jsonl",
    "API_URLS": ["http://10.80.12.34:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "generate_multi_turn_training_data",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/med_sample_20k_clear.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/degrade_med_sample_20k_a3b_ins_2507.jsonl",
    "API_URLS": ["http://10.80.12.34:8012/v1/chat/completions"]
  },
  {
    "STRATEGY": "strategy_direct_answer_and_correct",
    "INPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/ori_data/med_sample_20k_clear.jsonl",
    "OUTPUT_FILE": "/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/data/train_data/ori_med_sample_20k_a3b_ins_2507.jsonl",
    "API_URLS": ["http://10.80.12.34:8012/v1/chat/completions"]
  }
]'


# --- 组合队列以进行并行执行 ---
# 在下面的列表中添加或删除你想运行的队列变量。
# 例如，要只运行第一个队列，可以写成: PARALLEL_QUEUES=( "$SERIAL_QUEUE_1" )
# 下面的配置会并行运行以上定义的三个队列。
PARALLEL_QUEUES=(
  "$SERIAL_QUEUE_1"
  "$SERIAL_QUEUE_2"
)


# ==============================================================================
# --- 执行脚本 (无需修改) ---
# ==============================================================================
# 这部分代码会自动将上面定义的队列组合成 Python 脚本所需的最终 JSON 格式。

# 检查是否有任何队列被定义
if [ ${#PARALLEL_QUEUES[@]} -eq 0 ]; then
  echo "警告: PARALLEL_QUEUES 数组为空，没有任务需要执行。"
  exit 0
fi

# 使用逗号连接所有队列，并用方括号包围，以形成一个有效的JSON数组
TASK_CONFIG=$(IFS=,; echo "[${PARALLEL_QUEUES[*]}]")

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "任务配置加载完毕，准备启动 Python 调度器..."
echo "将在目录: $DIR 中执行"
echo "将并行执行 ${#PARALLEL_QUEUES[@]} 个任务队列。"
echo "--------------------------------------------------"

# 使用 python3 执行调度器，并将任务配置作为命令行参数传递。
# (cd ... && python3 ...) 确保脚本在正确的目录上下文中运行。
(cd "$DIR" && python3 main_queue.py "$TASK_CONFIG")

echo "--------------------------------------------------"
echo "Shell 脚本执行完毕。"
