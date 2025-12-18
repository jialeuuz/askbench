#!/bin/bash
export PYTHONPATH='/Users/zhaojiale5/local_codes/askQ/ask_eval'
# 如果任何命令失败，则立即退出
set -e

# --- 默认文件路径 (相对于项目根目录) ---
BASE_CONFIG_PATH="config/base.ini"
MAIN_PY_PATH="scripts/main.py"
DEFAULT_RESULTS_ROOT="results"

# --- 参数配置与说明 ---
# 直接在本脚本中修改以下变量，将覆盖 base.ini 中的对应值。

# [可选] 模型 sk_token
MODEL_SK_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJzbFhwYnpMOHRHenhnY2dFdFh4azgxMzVIdUhuSGlZYiJ9.e6PbiPCLNvBoGDcbZmHYiWsk6VE9b9tvmoCoT_zVM4U"
# [必需] 模型的 API URL
API_URL="http://api-hub.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME="azure-gpt-4_1"
# MODEL_NAME="gemini-2_5-pro"
# API_URL="http://10.80.12.180:8012/v1"
# MODEL_NAME="default"
# [必需] 逗号分隔的任务列表
# math500,medqa,gpqa,bbh
# quest_bench,in3_interaction
# ask_mind,ask_overconfidence (四个子集各采样100条，共400)
# ask_mind_math500de,ask_mind_medqade,ask_mind_gpqade,ask_mind_bbhde
# ask_overconfidence_math500,ask_overconfidence_medqa,ask_overconfidence_gpqa,ask_overconfidence_bbh
# healthbench
TASKS="healthbench"
# [可选] 手动指定结果保存目录。若不指定，将根据模型和任务自动生成。
SAVE_DIR="results/test-gpt-4_1"
# SAVE_DIR="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/results/qwen3_8b"
# [可选] [evaluatorconfig] api_url
EVAL_API_URL="http://api-hub.inner.chj.cloud/llm-gateway/v1"
# [可选] 裁判模型 sk_token
EVAL_SK_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJzbFhwYnpMOHRHenhnY2dFdFh4azgxMzVIdUhuSGlZYiJ9.e6PbiPCLNvBoGDcbZmHYiWsk6VE9b9tvmoCoT_zVM4U"
# [可选] 裁判模型名字
EVAL_MODEL_NAME="azure-gpt-4_1"
# EVAL_API_URL="https://lisunzhu123.fc.chj.cloud/gpt_41"
# [可选] [generateconfig] max_tokens
MAX_TOKENS=16000
# [可选] [generateconfig] temperature
TEMPERATURE="0.7"
# [可选] [generateconfig] max_concurrent
GEN_MAX_CONCURRENT=100
# [可选] [evaluatorconfig] max_concurrent
EVAL_MAX_CONCURRENT=100
# [可选] 首轮引导模式：none/weak/strong/fata
GUIDANCE_MODE="none"
# [可选] AskBench 严格模式：0/1（默认 0）。开启后将启用更严格的 Judge 判定与两轮流程约束。
STRICT_MODE="0"

# --- 校验必需参数 ---
if [ -z "${API_URL}" ] || [ -z "${TASKS}" ]; then
    echo "错误: 请在 run.sh 顶部配置 API_URL 与 TASKS。"
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
echo "将覆盖以下 base.ini 参数:"
# <-- 修改点4: 更新打印信息，使其更清晰，并使用新变量名
[ ! -z "${EVAL_API_URL}" ]        && echo "  [evaluatorconfig] api_url: ${EVAL_API_URL}"
[ ! -z "${MODEL_SK_TOKEN}" ]      && echo "  [model] sk_token:            ****(hidden)"
[ ! -z "${EVAL_SK_TOKEN}" ]       && echo "  [evaluatorconfig] sk_token:  ****(hidden)"
[ ! -z "${MODEL_NAME}" ]          && echo "  [model] model_name:         ${MODEL_NAME}"
[ ! -z "${EVAL_MODEL_NAME}" ]     && echo "  [evaluatorconfig] model_name:${EVAL_MODEL_NAME}"
[ ! -z "${MAX_TOKENS}" ]          && echo "  [generateconfig] max_tokens:         ${MAX_TOKENS}"
[ ! -z "${TEMPERATURE}" ]         && echo "  [generateconfig] temperature:        ${TEMPERATURE}"
[ ! -z "${GEN_MAX_CONCURRENT}" ]  && echo "  [generateconfig] max_concurrent: ${GEN_MAX_CONCURRENT}"
[ ! -z "${EVAL_MAX_CONCURRENT}" ] && echo "  [evaluatorconfig] max_concurrent:${EVAL_MAX_CONCURRENT}"
[ ! -z "${GUIDANCE_MODE}" ]       && echo "  [ask_evaluator] guidance_mode:       ${GUIDANCE_MODE}"
[ ! -z "${STRICT_MODE}" ]         && echo "  [ask_evaluator] strict_mode:         ${STRICT_MODE}"
echo "---"

# --- 修改配置文件 ---
echo "正在修改配置文件: ${BASE_CONFIG_PATH}"
cp "${BASE_CONFIG_PATH}" "${BASE_CONFIG_PATH}.bak"

update_config() {
    local section="$1"
    local key="$2"
    local value="$3"
    # 使用 awk 安全地更新 ini 文件；若键不存在会自动追加到该 section 尾部
    awk -v s="[${section}]" -v k="${key}" -v v="${value}" '
        BEGIN {in_section=0; updated=0}
        $0 == s {
            in_section=1
            print
            next
        }
        /^\[/ {
            if (in_section && !updated) {
                print k " = " v
                updated=1
            }
            in_section = ($0 == s)
            print
            next
        }
        {
            if (in_section && $1 == k) {
                print k " = " v
                updated=1
            } else {
                print
            }
        }
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
[ ! -z "${MODEL_SK_TOKEN}" ]      && update_config "model" "sk_token" "${MODEL_SK_TOKEN}"
[ ! -z "${EVAL_SK_TOKEN}" ]       && update_config "evaluatorconfig" "sk_token" "${EVAL_SK_TOKEN}"
[ ! -z "${MODEL_NAME}" ]          && update_config "model" "model_name" "${MODEL_NAME}"
[ ! -z "${EVAL_MODEL_NAME}" ]     && update_config "evaluatorconfig" "model_name" "${EVAL_MODEL_NAME}"
[ ! -z "${MAX_TOKENS}" ]          && update_config "generateconfig" "max_tokens" "${MAX_TOKENS}"
[ ! -z "${TEMPERATURE}" ]         && update_config "generateconfig" "temperature" "${TEMPERATURE}"
[ ! -z "${GEN_MAX_CONCURRENT}" ]  && update_config "generateconfig" "max_concurrent" "${GEN_MAX_CONCURRENT}"
[ ! -z "${EVAL_MAX_CONCURRENT}" ] && update_config "evaluatorconfig" "max_concurrent" "${EVAL_MAX_CONCURRENT}"
[ "${STRICT_MODE}" == "1" ]       && update_config "evaluatorconfig" "max_turns" "2"

echo "配置文件修改完成。"
echo "---"

# --- 运行主程序 ---
echo "正在启动评估脚本: ${MAIN_PY_PATH}"
export GUIDANCE_MODE
export STRICT_MODE
python "${MAIN_PY_PATH}" --config "${BASE_CONFIG_PATH}"

# --- 恢复配置文件 ---
mv "${BASE_CONFIG_PATH}.bak" "${BASE_CONFIG_PATH}"

echo "---"
echo "评估任务执行完毕。"
echo "结果已保存至: ${SAVE_DIR}"
echo "配置文件 ${BASE_CONFIG_PATH} 已从备份中恢复。"
echo "---"
