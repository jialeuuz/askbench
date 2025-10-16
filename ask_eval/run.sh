#!/bin/bash
export PYTHONPATH='/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/bak/ask_eval'
# 如果任何命令失败，则立即退出
set -e

# --- 默认文件路径 (相对于项目根目录) ---
BASE_CONFIG_PATH="config/base.ini"
MAIN_PY_PATH="scripts/main.py"
DEFAULT_RESULTS_ROOT="results"

# --- 参数配置与说明 ---
# 通过命令行传入的参数将覆盖 base.ini 中的对应值。

# [必需] 模型的 API URL
API_URL="http://10.80.12.28:8013/v1/chat/completions"
# [必需] 逗号分隔的任务列表 (math500,medqa,aime2025,aime2025_de,math500_de,medqa_de,ask_yes,ask_mind,ask_lone)
# math500,medqa,aime2025,
# ask_mind_math500de,ask_mind_medqade,ask_mind_aime2025de
# quest_bench
TASKS="math500,medqa,aime2025,ask_mind_math500de,ask_mind_medqade,ask_mind_aime2025de,quest_bench"
# [可选] 手动指定结果保存目录。若不指定，将根据模型和任务自动生成。
SAVE_DIR="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/bak/results/qwen2.5_7b_step200_med_oss120_low"
# SAVE_DIR="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/results/qwen25_7b_ins"
# [可选] [evaluatorconfig] api_url
EVAL_API_URL="http://10.80.13.117:8012/v1/chat/completions,http://10.80.13.117:8013/v1/chat/completions,http://10.80.13.117:8014/v1/chat/completions,http://10.80.13.117:8015/v1/chat/completions"
# EVAL_API_URL="https://lisunzhu123.fc.chj.cloud/gpt_41"
# [可选] [generateconfig] max_tokens 
MAX_TOKENS=8000
# [可选] [generateconfig] temperature
TEMPERATURE="0.7"
# [可选] [generateconfig] max_concurrent
GEN_MAX_CONCURRENT=100
# [可选] [evaluatorconfig] max_concurrent
EVAL_MAX_CONCURRENT=50


# --- 解析命令行参数 ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--url)           API_URL="$2"; shift ;;
        -t|--tasks)         TASKS="$2"; shift ;;
        -s|--save-dir)      SAVE_DIR="$2"; shift ;;
        -e|--eval-api-url)  EVAL_API_URL="$2"; shift ;; # <-- 修改点3: 命令行参数改为 --eval-api-url
        --max-tokens)       MAX_TOKENS="$2"; shift ;;
        --temp)             TEMPERATURE="$2"; shift ;;
        --gen-concurrent)   GEN_MAX_CONCURRENT="$2"; shift ;;
        --eval-concurrent)  EVAL_MAX_CONCURRENT="$2"; shift ;;
        # 如果用户请求帮助，或者输入了未知参数，给一个简短的提示
        -h|--help|*)
            echo "错误或请求帮助。请直接阅读脚本开头的注释以了解用法。"
            echo "必需参数: -u <url>, -t <tasks>"
            exit 1
            ;;
    esac
    shift
done

# --- 校验必需参数 ---
if [ -z "${API_URL}" ] || [ -z "${TASKS}" ]; then
    echo "错误: 参数 -u (url) 和 -t (tasks) 是必需项。"
    echo "请直接阅读脚本开头的注释以了解用法。"
    exit 1
fi

# --- 自动生成 save_dir (如果未手动指定) ---
if [ -z "${SAVE_DIR}" ]; then
    MODEL_ID=$(echo "${API_URL}" | sed -n 's#.*//\([^/]*\).*#\1#p' | tr '.' '_' | tr ':' '_')
    TASK_SUFFIX=$(echo "${TASKS}" | tr ',' '_')
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    AUTO_SAVE_DIR_NAME="${MODEL_ID}_${TASK_SUFFIX}_${TIMESTAMP}"
    SAVE_DIR="${DEFAULT_RESULTS_ROOT}/${AUTO_SAVE_DIR_NAME}"
fi

# --- 打印执行配置 ---
echo "---"
echo "准备执行评估任务..."
echo "API URL:            ${API_URL}"
echo "任务列表:           ${TASKS}"
echo "结果保存目录:       ${SAVE_DIR}"
echo "---"
echo "将覆盖以下 base.ini 参数 (如果已提供):"
# <-- 修改点4: 更新打印信息，使其更清晰，并使用新变量名
[ ! -z "${EVAL_API_URL}" ]        && echo "  [evaluatorconfig] api_url: ${EVAL_API_URL}"
[ ! -z "${MAX_TOKENS}" ]          && echo "  [generateconfig] max_tokens:         ${MAX_TOKENS}"
[ ! -z "${TEMPERATURE}" ]         && echo "  [generateconfig] temperature:        ${TEMPERATURE}"
[ ! -z "${GEN_MAX_CONCURRENT}" ]  && echo "  [generateconfig] max_concurrent: ${GEN_MAX_CONCURRENT}"
[ ! -z "${EVAL_MAX_CONCURRENT}" ] && echo "  [evaluatorconfig] max_concurrent:${EVAL_MAX_CONCURRENT}"
echo "---"

# --- 修改配置文件 ---
echo "正在修改配置文件: ${BASE_CONFIG_PATH}"
cp "${BASE_CONFIG_PATH}" "${BASE_CONFIG_PATH}.bak"

update_config() {
    local section="$1"
    local key="$2"
    local value="$3"
    # 使用 awk 安全地更新 ini 文件
    awk -v s="[${section}]" -v k="${key}" -v v="${value}" '
        BEGIN {updated=0}
        $0 == s {in_section=1}
        /^\[/ && $0 != s {in_section=0}
        in_section && $1 == k {
            $0 = k " = " v;
            updated=1
        }
        {print}
        END {
            if (in_section && !updated) {
                print k " = " v
            }
        }
    ' "${BASE_CONFIG_PATH}" > "${BASE_CONFIG_PATH}.tmp" && mv "${BASE_CONFIG_PATH}.tmp" "${BASE_CONFIG_PATH}"
}

update_config "model" "api_url" "${API_URL}"
update_config "tasks" "enabled" "${TASKS}"
update_config "path" "save_dir" "${SAVE_DIR}"

# <-- 修改点5: 这是最核心的修改，将 "evaluator_url" 改为 "api_url"，并使用新变量
[ ! -z "${EVAL_API_URL}" ]        && update_config "evaluatorconfig" "api_url" "${EVAL_API_URL}"
[ ! -z "${MAX_TOKENS}" ]          && update_config "generateconfig" "max_tokens" "${MAX_TOKENS}"
[ ! -z "${TEMPERATURE}" ]         && update_config "generateconfig" "temperature" "${TEMPERATURE}"
[ ! -z "${GEN_MAX_CONCURRENT}" ]  && update_config "generateconfig" "max_concurrent" "${GEN_MAX_CONCURRENT}"
[ ! -z "${EVAL_MAX_CONCURRENT}" ] && update_config "evaluatorconfig" "max_concurrent" "${EVAL_MAX_CONCURRENT}"

echo "配置文件修改完成。"
echo "---"

# --- 运行主程序 ---
echo "正在启动评估脚本: ${MAIN_PY_PATH}"
python "${MAIN_PY_PATH}" --config "${BASE_CONFIG_PATH}"

# --- 恢复配置文件 ---
mv "${BASE_CONFIG_PATH}.bak" "${BASE_CONFIG_PATH}"

echo "---"
echo "评估任务执行完毕。"
echo "结果已保存至: ${SAVE_DIR}"
echo "配置文件 ${BASE_CONFIG_PATH} 已从备份中恢复。"
echo "---"