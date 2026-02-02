#!/bin/bash
set -e
MODEL_PATH=/Qwen2.5-7B-Instruct

PORT=8012
# CUDA_DEVICES="0,1,2,3,4,5,6,7"
# CUDA_DEVICES="0,1"
# CUDA_DEVICES="2,3"
# CUDA_DEVICES="4,5"
# CUDA_DEVICES="6,7"
CUDA_DEVICES="0,1,2,3"
# CUDA_DEVICES="4,5,6,7"
TP=4  # Tensor-parallel size (should match the number of GPUs in CUDA_DEVICES)

echo "Using MODEL_PATH=${MODEL_PATH}"
echo "CUDA_DEVICES=${CUDA_DEVICES}"
echo "Tensor Parallel Size=${TP}"

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=eth0

# Optional: set a writable cache dir for tiktoken-rs (leave empty to use default).
# Example:
# TIKTOKEN_RS_CACHE_DIR="/tmp/tiktoken-cache"
if [ -n "${TIKTOKEN_RS_CACHE_DIR:-}" ]; then
    export TIKTOKEN_RS_CACHE_DIR
fi


python -m vllm.entrypoints.openai.api_server \
    --served-model-name default \
    --model ${MODEL_PATH} \
    --tensor-parallel-size ${TP} \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1024 \
    --host 0.0.0.0 \
    --port ${PORT} \
    --enable-log-requests
