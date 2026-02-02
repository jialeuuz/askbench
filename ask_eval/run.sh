#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
# Exit immediately if any command fails.
set -e

# --- Default file paths (relative to the project root) ---
BASE_CONFIG_PATH="config/base.ini"
MAIN_PY_PATH="scripts/main.py"
DEFAULT_RESULTS_ROOT="results"

# --- Parameters and notes ---
# Editing variables in this script overrides the corresponding values in base.ini.

# [Optional] Model auth token (set via env var, or fill here)
MODEL_SK_TOKEN="${MODEL_SK_TOKEN:-}"
# [Required] Candidate model API URL
# API_URL="http://host:port/v1"
# MODEL_NAME="azure-gpt-4_1"
# MODEL_NAME="gemini-2_5-pro"
API_URL="http://host:port/v1/chat/completions"
MODEL_NAME="default"
# [Required] Comma-separated task list
# math500,medqa,gpqa,bbh
# quest_bench,in3_interaction
# ask_mind,ask_overconfidence (100 sampled from each subset, 400 total)
# ask_mind_math500de,ask_mind_medqade,ask_mind_gpqade,ask_mind_bbhde
# ask_overconfidence_math500,ask_overconfidence_medqa,ask_overconfidence_gpqa,ask_overconfidence_bbh
# healthbench
TASKS="healthbench"
# [Optional] Manually set the output directory. If empty, it will be auto-generated.
SAVE_DIR=""
# Example:
# SAVE_DIR="/path/to/save_dir"
# [Optional] [evaluatorconfig] api_url
# EVAL_API_URL="http://host:port/v1"
# EVAL_MODEL_NAME="azure-gpt-4_1"
# [Optional] Judge model auth token (set via env var, or fill here)
EVAL_SK_TOKEN="${EVAL_SK_TOKEN:-}"
EVAL_MODEL_NAME="default"
EVAL_API_URL="http://host:port/v1/chat/completions"
# [Optional] [generateconfig] max_tokens
MAX_TOKENS=16000
# [Optional] [generateconfig] temperature
TEMPERATURE="0.7"
# [Optional] [generateconfig] max_concurrent
GEN_MAX_CONCURRENT=100
# [Optional] [evaluatorconfig] max_concurrent
EVAL_MAX_CONCURRENT=100
# [Optional] First-turn guidance mode: none/weak/strong/fata
GUIDANCE_MODE="none"
# [Optional] AskBench strict mode: 0/1 (default 0). When enabled, uses stricter judge rules and enforces a two-turn protocol.
STRICT_MODE="0"
# [Optional] Max dialogue turns (default 3). Override via CLI: ./run.sh --max-turns 4
MAX_TURNS="3"

print_usage() {
    cat <<EOF
Usage: ./run.sh [--max-turns N]

Options:
  --max-turns, -t   Set evaluator max_turns (default: 3)
  --help, -h        Show this help
EOF
}

# --- CLI args (higher priority than in-script variables) ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-turns|-t)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --max-turns requires an integer argument" >&2
                exit 1
            fi
            MAX_TURNS="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if ! [[ "${MAX_TURNS}" =~ ^[0-9]+$ ]] || [[ "${MAX_TURNS}" -lt 1 ]]; then
    echo "Error: --max-turns must be an integer >= 1 (got: ${MAX_TURNS})" >&2
    exit 1
fi

# STRICT_MODE forces a 2-turn protocol (aligned with the evaluator’s internal logic).
if [[ "${STRICT_MODE}" == "1" ]] && [[ "${MAX_TURNS}" != "2" ]]; then
    echo "[STRICT_MODE] Overriding --max-turns: ${MAX_TURNS} -> 2" >&2
    MAX_TURNS="2"
fi

# --- Validate required args ---
if [ -z "${API_URL}" ] || [ -z "${TASKS}" ]; then
    echo "Error: please configure API_URL and TASKS at the top of run.sh."
    exit 1
fi

# --- Auto-generate save_dir (if not set) ---
if [ -z "${SAVE_DIR}" ]; then
    MODEL_ID=$(echo "${API_URL}" | sed -n 's#.*//\([^/]*\).*#\1#p' | tr '.' '_' | tr ':' '_')
    TASK_SUFFIX=$(echo "${TASKS}" | tr ',' '_')
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    AUTO_SAVE_DIR_NAME="${MODEL_ID}_${TASK_SUFFIX}_${TIMESTAMP}"
    SAVE_DIR="${DEFAULT_RESULTS_ROOT}/${AUTO_SAVE_DIR_NAME}"
fi

# --- Print run config ---
echo "---"
echo "Preparing evaluation run..."
echo "API URL:            ${API_URL}"
echo "Tasks:              ${TASKS}"
echo "Output dir:          ${SAVE_DIR}"
echo "---"
echo "Will override these base.ini keys:"
# Note: keep the output readable; hide tokens.
[ ! -z "${EVAL_API_URL}" ]        && echo "  [evaluatorconfig] api_url: ${EVAL_API_URL}"
[ ! -z "${MODEL_SK_TOKEN}" ]      && echo "  [model] sk_token:            ****(hidden)"
[ ! -z "${EVAL_SK_TOKEN}" ]       && echo "  [evaluatorconfig] sk_token:  ****(hidden)"
[ ! -z "${MODEL_NAME}" ]          && echo "  [model] model_name:         ${MODEL_NAME}"
[ ! -z "${EVAL_MODEL_NAME}" ]     && echo "  [evaluatorconfig] model_name:${EVAL_MODEL_NAME}"
[ ! -z "${MAX_TOKENS}" ]          && echo "  [generateconfig] max_tokens:         ${MAX_TOKENS}"
[ ! -z "${TEMPERATURE}" ]         && echo "  [generateconfig] temperature:        ${TEMPERATURE}"
[ ! -z "${GEN_MAX_CONCURRENT}" ]  && echo "  [generateconfig] max_concurrent: ${GEN_MAX_CONCURRENT}"
[ ! -z "${EVAL_MAX_CONCURRENT}" ] && echo "  [evaluatorconfig] max_concurrent:${EVAL_MAX_CONCURRENT}"
[ ! -z "${MAX_TURNS}" ]           && echo "  [evaluatorconfig] max_turns:     ${MAX_TURNS}"
[ ! -z "${GUIDANCE_MODE}" ]       && echo "  [ask_evaluator] guidance_mode:       ${GUIDANCE_MODE}"
[ ! -z "${STRICT_MODE}" ]         && echo "  [ask_evaluator] strict_mode:         ${STRICT_MODE}"
echo "---"

# --- Update config file ---
echo "Updating config file: ${BASE_CONFIG_PATH}"
cp "${BASE_CONFIG_PATH}" "${BASE_CONFIG_PATH}.bak"

update_config() {
    local section="$1"
    local key="$2"
    local value="$3"
    # Safely update an INI key via awk. If the key does not exist, append it to the end of the section.
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

# Core change: update evaluator key name from "evaluator_url" to "api_url".
[ ! -z "${EVAL_API_URL}" ]        && update_config "evaluatorconfig" "api_url" "${EVAL_API_URL}"
[ ! -z "${MODEL_SK_TOKEN}" ]      && update_config "model" "sk_token" "${MODEL_SK_TOKEN}"
[ ! -z "${EVAL_SK_TOKEN}" ]       && update_config "evaluatorconfig" "sk_token" "${EVAL_SK_TOKEN}"
[ ! -z "${MODEL_NAME}" ]          && update_config "model" "model_name" "${MODEL_NAME}"
[ ! -z "${EVAL_MODEL_NAME}" ]     && update_config "evaluatorconfig" "model_name" "${EVAL_MODEL_NAME}"
[ ! -z "${MAX_TOKENS}" ]          && update_config "generateconfig" "max_tokens" "${MAX_TOKENS}"
[ ! -z "${TEMPERATURE}" ]         && update_config "generateconfig" "temperature" "${TEMPERATURE}"
[ ! -z "${GEN_MAX_CONCURRENT}" ]  && update_config "generateconfig" "max_concurrent" "${GEN_MAX_CONCURRENT}"
[ ! -z "${EVAL_MAX_CONCURRENT}" ] && update_config "evaluatorconfig" "max_concurrent" "${EVAL_MAX_CONCURRENT}"
[ ! -z "${MAX_TURNS}" ]           && update_config "evaluatorconfig" "max_turns" "${MAX_TURNS}"

echo "Config update done."
echo "---"

# --- Run ---
echo "Launching evaluation script: ${MAIN_PY_PATH}"
export GUIDANCE_MODE
export STRICT_MODE
python "${MAIN_PY_PATH}" --config "${BASE_CONFIG_PATH}"

# --- Restore config file ---
mv "${BASE_CONFIG_PATH}.bak" "${BASE_CONFIG_PATH}"

echo "---"
echo "Evaluation finished."
echo "Results saved to: ${SAVE_DIR}"
echo "Config file ${BASE_CONFIG_PATH} restored from backup."
echo "---"
