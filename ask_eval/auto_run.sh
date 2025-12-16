#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}"

CONFIG_PATH="${ROOT_DIR}/auto_run.json"
DRY_RUN=0
ONLY_REGEX=""

# ---- 固定配置（一般只需要改这里）----
# Judge 模型（固定端口，不写进 JSON）
# vLLM OpenAI server 建议填完整 endpoint：/v1/chat/completions（可用逗号分隔多个 URL 做负载均衡）
JUDGE_API_URL="http://10.80.12.180:8014/v1/chat/completions"
JUDGE_API_TYPE="default"
JUDGE_MODEL_NAME="default"
JUDGE_SK_TOKEN="none"
JUDGE_TIMEOUT="600"
JUDGE_MAX_CONCURRENT="100"
JUDGE_TEMPERATURE="0.0"
JUDGE_MAX_NEW_TOKENS="2048"

# base.ini 模板与 python
BASE_INI_REL="config/base.ini"
PYTHON_BIN="python"

# vLLM 启动相关（一般不需要动）
API_HOST="127.0.0.1"
VLLM_BIND_HOST="0.0.0.0"
VLLM_SERVED_MODEL_NAME="default"
VLLM_GPU_MEMORY_UTILIZATION="0.9"
VLLM_MAX_NUM_SEQS="1024"
VLLM_TRUST_REMOTE_CODE="1"
VLLM_ENABLE_LOG_REQUESTS="1"

# 评测相关（一般不需要动）
GUIDANCE_MODE="none"
RESULTS_ROOT_REL="results/auto_run"
MODEL_API_TYPE="default"
MODEL_MODEL_NAME="default"
MODEL_SK_TOKEN="none"
MODEL_TIMEOUT=""
GEN_MAX_TOKENS="16000"
GEN_TEMPERATURE="0.7"
GEN_MAX_CONCURRENT="100"

# vLLM 启动前环境变量（按需改）
NCCL_P2P_DISABLE="1"
NCCL_IB_DISABLE="1"
NCCL_DEBUG="info"
NCCL_SOCKET_IFNAME="eth0"
TIKTOKEN_RS_CACHE_DIR="/mnt/pfs-guan-ssai/nlu/zhaojiale/models/data_generate"

usage() {
  cat <<'EOF'
Usage:
  ./auto_run.sh
  ./auto_run.sh [--dry-run] [--only <regex>]
  ./auto_run.sh -c <config.json> [--dry-run] [--only <regex>]

Notes:
  - Default config path: ./auto_run.json (copy from ./auto_run.example.json)
  - Config file is a JSON list (or {"models": [...]}) containing only eval model entries.
  - Judge/vLLM defaults are configured at the top of auto_run.sh (not in JSON).
  - This script reads the model list, deploys each eval model via vLLM on its configured port,
    then runs `scripts/main.py` with a per-run temp ini (does NOT modify `config/base.ini`).
  - Models on different ports run in parallel (one worker per port); models sharing a port run sequentially.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONFIG_PATH="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY_REGEX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -f "${CONFIG_PATH}" ]] || {
  echo "Config not found: ${CONFIG_PATH}" >&2
  echo "Create it by copying: ${ROOT_DIR}/auto_run.example.json -> ${ROOT_DIR}/auto_run.json" >&2
  exit 1
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
need_cmd python
need_cmd curl

RUN_ID="$(date +%Y%m%d-%H%M%S)_$$"
RUN_ROOT="${ROOT_DIR}/temp/auto_run/${RUN_ID}"
mkdir -p "${RUN_ROOT}"

update_ini() {
  local file="$1"
  local section="$2"
  local key="$3"
  local value="$4"
  awk -v s="[${section}]" -v k="${key}" -v v="${value}" '
      BEGIN {in_section=0; updated=0}
      $0 == s {in_section=1; print; next}
      /^\[/ {
          if (in_section && !updated) { print k " = " v; updated=1 }
          in_section = ($0 == s)
          print
          next
      }
      {
          if (in_section && $1 == k) { print k " = " v; updated=1 }
          else { print }
      }
      END { if (in_section && !updated) { print k " = " v } }
  ' "${file}" > "${file}.tmp" && mv "${file}.tmp" "${file}"
}

wait_for_vllm() {
  local api_host="$1"
  local port="$2"
  local timeout_s="$3"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if curl -sS "http://${api_host}:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > timeout_s )); then
      return 1
    fi
    sleep 2
  done
}

stop_pid_tree() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    for _ in {1..30}; do
      kill -0 "${pid}" >/dev/null 2>&1 || return 0
      sleep 1
    done
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
}

decode_model_to_env() {
  local b64="$1"
  python - "$b64" <<'PY'
import base64, json, shlex, sys
cfg = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
def emit(k, v):
    if v is None:
        return
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    else:
        v = str(v)
    print(f"{k}={shlex.quote(v)}")

emit("NAME", cfg.get("name"))
emit("MODEL_PATH", cfg.get("model_path"))
emit("PORT", cfg.get("port"))
emit("CUDA_DEVICES", cfg.get("cuda_devices"))
emit("TP", cfg.get("tensor_parallel_size"))
emit("TASKS_CSV", ",".join(cfg.get("tasks") or []))
emit("SAVE_DIR", cfg.get("save_dir"))
PY
}

run_one_model() {
  local model_b64="$1"
  eval "$(decode_model_to_env "${model_b64}")"

  if [[ "${SAVE_DIR}" != /* ]]; then
    SAVE_DIR="${ROOT_DIR}/${SAVE_DIR}"
  fi
  local base_ini="${BASE_INI_REL}"
  if [[ "${base_ini}" != /* ]]; then
    base_ini="${ROOT_DIR}/${base_ini}"
  fi

  if [[ -n "${ONLY_REGEX}" ]]; then
    if ! [[ "${NAME}" =~ ${ONLY_REGEX} ]]; then
      echo "[skip] ${NAME} (does not match --only)"
      return 0
    fi
  fi

  echo "---"
  echo "[model] ${NAME}"
  echo "  port=${PORT} cuda=${CUDA_DEVICES} tp=${TP}"
  echo "  api_url=http://${API_HOST}:${PORT}/v1/chat/completions"
  echo "  tasks=${TASKS_CSV}"
  echo "  save_dir=${SAVE_DIR}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  mkdir -p "${SAVE_DIR}"

  local work_dir="${RUN_ROOT}/port_${PORT}/${NAME}"
  mkdir -p "${work_dir}"
  local vllm_log="${work_dir}/vllm.log"
  local eval_log="${work_dir}/eval.log"

  local cfg_ini="${work_dir}/base.ini"
  cp "${base_ini}" "${cfg_ini}"

  update_ini "${cfg_ini}" "path" "save_dir" "${SAVE_DIR}"
  update_ini "${cfg_ini}" "tasks" "enabled" "${TASKS_CSV}"

  update_ini "${cfg_ini}" "model" "api_url" "http://${API_HOST}:${PORT}/v1/chat/completions"
  update_ini "${cfg_ini}" "model" "api_type" "${MODEL_API_TYPE}"
  update_ini "${cfg_ini}" "model" "model_name" "${MODEL_MODEL_NAME}"
  update_ini "${cfg_ini}" "model" "sk_token" "${MODEL_SK_TOKEN}"
  [[ -n "${MODEL_TIMEOUT}" ]] && update_ini "${cfg_ini}" "model" "timeout" "${MODEL_TIMEOUT}"

  [[ -n "${GEN_MAX_TOKENS}" ]] && update_ini "${cfg_ini}" "generateconfig" "max_tokens" "${GEN_MAX_TOKENS}"
  [[ -n "${GEN_TEMPERATURE}" ]] && update_ini "${cfg_ini}" "generateconfig" "temperature" "${GEN_TEMPERATURE}"
  [[ -n "${GEN_MAX_CONCURRENT}" ]] && update_ini "${cfg_ini}" "generateconfig" "max_concurrent" "${GEN_MAX_CONCURRENT}"

  update_ini "${cfg_ini}" "evaluatorconfig" "api_url" "${JUDGE_API_URL}"
  update_ini "${cfg_ini}" "evaluatorconfig" "api_type" "${JUDGE_API_TYPE}"
  update_ini "${cfg_ini}" "evaluatorconfig" "model_name" "${JUDGE_MODEL_NAME}"
  update_ini "${cfg_ini}" "evaluatorconfig" "sk_token" "${JUDGE_SK_TOKEN}"
  [[ -n "${JUDGE_TIMEOUT}" ]] && update_ini "${cfg_ini}" "evaluatorconfig" "timeout" "${JUDGE_TIMEOUT}"
  [[ -n "${JUDGE_MAX_CONCURRENT}" ]] && update_ini "${cfg_ini}" "evaluatorconfig" "max_concurrent" "${JUDGE_MAX_CONCURRENT}"
  [[ -n "${JUDGE_TEMPERATURE}" ]] && update_ini "${cfg_ini}" "evaluatorconfig" "temperature" "${JUDGE_TEMPERATURE}"
  [[ -n "${JUDGE_MAX_NEW_TOKENS}" ]] && update_ini "${cfg_ini}" "evaluatorconfig" "max_new_tokens" "${JUDGE_MAX_NEW_TOKENS}"

  local vllm_pid=""
  cleanup() {
    stop_pid_tree "${vllm_pid}"
  }
  trap cleanup EXIT

  local trust_flag=()
  [[ "${VLLM_TRUST_REMOTE_CODE}" == "1" ]] && trust_flag=(--trust-remote-code)
  local log_req_flag=()
  [[ "${VLLM_ENABLE_LOG_REQUESTS}" == "1" ]] && log_req_flag=(--enable-log-requests)

  echo "[deploy] starting vLLM on :${PORT} ..."
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
    export NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_DEBUG NCCL_SOCKET_IFNAME TIKTOKEN_RS_CACHE_DIR
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
      --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
      --model "${MODEL_PATH}" \
      --tensor-parallel-size "${TP}" \
      "${trust_flag[@]}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --host "${VLLM_BIND_HOST}" \
      --port "${PORT}" \
      "${log_req_flag[@]}" \
      >"${vllm_log}" 2>&1 &
    echo $! > "${work_dir}/vllm.pid"
  )
  vllm_pid="$(cat "${work_dir}/vllm.pid")"

  if ! wait_for_vllm "${API_HOST}" "${PORT}" 900; then
    echo "[deploy] vLLM did not become ready in time. tail -n 80 ${vllm_log}:" >&2
    tail -n 80 "${vllm_log}" >&2 || true
    return 1
  fi

  echo "[eval] running scripts/main.py ..."
  (
    export GUIDANCE_MODE="${GUIDANCE_MODE}"
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/main.py" --config "${cfg_ini}" >"${eval_log}" 2>&1
  )
  echo "[eval] done: ${NAME}"

  cleanup
  trap - EXIT
  return 0
}

run_worker_for_port() {
  local port="$1"
  local queue_file="$2"
  echo "[worker:${port}] queue=${queue_file}"
  local line
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    run_one_model "${line}"
  done < "${queue_file}"
}

echo "[config] ${CONFIG_PATH}"
echo "[run_root] ${RUN_ROOT}"

# Expand config into per-model base64 blobs, grouped by port.
python - "${CONFIG_PATH}" "${ONLY_REGEX}" "${RESULTS_ROOT_REL}" <<'PY' > "${RUN_ROOT}/models.tsv"
import base64, json, os, re, sys

config_path = sys.argv[1]
only_regex = sys.argv[2] if len(sys.argv) > 2 else ""
results_root = sys.argv[3] if len(sys.argv) > 3 else "results/auto_run"

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

if isinstance(cfg, list):
    models = cfg
elif isinstance(cfg, dict):
    models = cfg.get("models") or []
else:
    raise SystemExit("Config must be a JSON list or an object with 'models'")

def to_tasks(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [t.strip() for t in x.split(",") if t.strip()]
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    raise ValueError("tasks must be string or list")

if not models:
    raise SystemExit("No models found in config.models")

# 固定端口映射（你只用 8012/8013 就够了；也可在模型里用 cuda_devices 覆盖）
PORT_TO_CUDA = {8012: "0,1", 8013: "2,3"}

for m in models:
    model_path = m.get("model_path")
    port = m.get("port")
    if not model_path or port is None:
        raise SystemExit("Each model requires model_path and port")
    port = int(port)

    name = m.get("name") or os.path.basename(str(model_path).rstrip("/")) or f"model_{port}"
    if only_regex and not re.search(only_regex, name):
        continue

    cuda_devices = m.get("cuda_devices") or PORT_TO_CUDA.get(port) or ""
    if not cuda_devices:
        raise SystemExit(f"Missing cuda_devices for port {port}; set models[].cuda_devices (e.g., \"0,1\")")

    tp = m.get("tp") or m.get("tensor_parallel_size")
    if tp is None:
        tp = len([x for x in str(cuda_devices).split(",") if x.strip()]) or 1
    tp = int(tp)

    tasks = to_tasks(m.get("tasks"))
    if not tasks:
        raise SystemExit(f"Model {name} has empty tasks")

    save_dir = m.get("save_dir") or os.path.join(results_root, name)

    merged = {
        "name": name,
        "model_path": model_path,
        "port": port,
        "cuda_devices": cuda_devices,
        "tensor_parallel_size": tp,
        "tasks": tasks,
        "save_dir": save_dir,
    }

    blob = base64.b64encode(json.dumps(merged, ensure_ascii=False).encode("utf-8")).decode("ascii")
    print(f"{port}\t{blob}")
PY

if [[ ! -s "${RUN_ROOT}/models.tsv" ]]; then
  echo "No models matched config/filters." >&2
  exit 1
fi

declare -a PORTS=()

while IFS=$'\t' read -r port blob; do
  [[ -n "${port}" && -n "${blob}" ]] || continue
  q="${RUN_ROOT}/queue_${port}.txt"
  if [[ ! -f "${q}" ]]; then
    PORTS+=("${port}")
    : > "${q}"
  fi
  echo "${blob}" >> "${q}"
done < "${RUN_ROOT}/models.tsv"

if (( ${#PORTS[@]} == 0 )); then
  echo "No models matched config/filters." >&2
  exit 1
fi
echo "[ports] ${PORTS[*]}"

declare -a WORKER_PIDS=()
declare -a WORKER_PORTS=()
for port in "${PORTS[@]}"; do
  run_worker_for_port "${port}" "${RUN_ROOT}/queue_${port}.txt" &
  WORKER_PORTS+=("${port}")
  WORKER_PIDS+=("$!")
done

exit_code=0
for idx in "${!WORKER_PIDS[@]}"; do
  pid="${WORKER_PIDS[$idx]}"
  port="${WORKER_PORTS[$idx]}"
  if ! wait "${pid}"; then
    echo "[worker:${port}] failed" >&2
    exit_code=1
  fi
done

echo "---"
echo "All done. Logs and temp configs: ${RUN_ROOT}"
exit "${exit_code}"
