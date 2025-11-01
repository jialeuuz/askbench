#!/bin/bash
MODEL_PATH=/lpai/dataset/rubrics-models/0-2-0/Qwen2.5-7B-Instruct

PORT=8013
# CUDA_DEVICES="0,1,2,3,4,5,6,7"
# CUDA_DEVICES="0,1"
CUDA_DEVICES="2,3"
# CUDA_DEVICES="4,5"
# CUDA_DEVICES="6,7"
# CUDA_DEVICES="0,1,2,3"
# CUDA_DEVICES="4,5,6,7"
TP=2  # 等于你想用的张量并行大小，和设备数量一致

echo "Using MODEL_PATH=${MODEL_PATH}"
echo "CUDA_DEVICES=${CUDA_DEVICES}"
echo "Tensor Parallel Size=${TP}"

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=eth0

export TIKTOKEN_RS_CACHE_DIR=/mnt/pfs-guan-ssai/nlu/zhaojiale/models/data_generate


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
