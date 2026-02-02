#!/bin/bash
set -euo pipefail

# Reference training script (paper setup).
# This repo does NOT vendor VERL; clone it separately and set VERL_DIR.
#
# Example:
#   export VERL_DIR="/path/to/verl"
#   export MODEL_PATH="/path/to/hf_model"
#   export DATA_TRAIN_PATH="/path/to/train.parquet"
#   export DATA_VAL_PATH="/path/to/val.parquet"
#   export OUTPUT_DIR="/path/to/output_dir"
#   bash reward/train.sh

VERL_DIR="${VERL_DIR:-/path/to/verl}"
if [ ! -d "${VERL_DIR}" ]; then
    echo "❌ VERL_DIR not found: ${VERL_DIR}" >&2
    echo "Set VERL_DIR to your local VERL repo root (see reward/readme)." >&2
    exit 1
fi

# -----------------------------
# Experiment / paths (edit me)
# -----------------------------
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
DATA_NAME="${DATA_NAME:-ask_mind_example}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_NAME}_${DATA_NAME}}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXPERIMENT_NAME}}"
MODEL_PATH="${MODEL_PATH:-/path/to/hf_model}"
DATA_TRAIN_PATH="${DATA_TRAIN_PATH:-/path/to/train.parquet}"
DATA_VAL_PATH="${DATA_VAL_PATH:-/path/to/val.parquet}"

mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ MODEL_PATH not found (expected a HuggingFace model dir): ${MODEL_PATH}" >&2
    exit 1
fi
if [ ! -f "${DATA_TRAIN_PATH}" ]; then
    echo "❌ DATA_TRAIN_PATH not found: ${DATA_TRAIN_PATH}" >&2
    exit 1
fi
if [ ! -f "${DATA_VAL_PATH}" ]; then
    echo "❌ DATA_VAL_PATH not found: ${DATA_VAL_PATH}" >&2
    exit 1
fi

# -----------------------------
# Ray
# -----------------------------
export RAY_num_server_call_thread="${RAY_num_server_call_thread:-1}"
export RAY_TMPDIR="${RAY_TMPDIR:-${OUTPUT_DIR}/ray_tmp}"
mkdir -p "${RAY_TMPDIR}"

RAY_PORT="${RAY_PORT:-7801}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-7802}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"

cd "${VERL_DIR}"
pip install -e .

ray start --head --port="${RAY_PORT}" --dashboard-port="${RAY_DASHBOARD_PORT}" --num-cpus="${RAY_NUM_CPUS}"

#=============================


# Model configuration
# Experiment configuration
ROLLOUT_LOG_DIR="${OUTPUT_DIR}/rollout_log/${EXPERIMENT_NAME}"
VALIDATION_LOG_DIR="${OUTPUT_DIR}/validation_log/${EXPERIMENT_NAME}"


# Dynamic batch configuration
max_prompt_length=2048
max_response_length=8192
use_dynamic_bsz=True
max_tokens=$((max_prompt_length + max_response_length))
max_tokens=$((1024 * 22))
actor_ppo_max_token_len=$((max_tokens * 2))
infer_ppo_max_token_len=$((max_tokens * 3))
max_num_batched_tokens=$((max_tokens * 3))

# DAPO specific parameters
clip_ratio_low=0.2
clip_ratio_high=0.28

export WANDB_MODE="${WANDB_MODE:-offline}"
set -x

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_TRAIN_PATH}" \
    data.val_files="${DATA_VAL_PATH}" \
    data.train_batch_size=64 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.rollout_data_dir="${ROLLOUT_LOG_DIR}" \
    trainer.validation_data_dir="${VALIDATION_LOG_DIR}" \
    trainer.total_training_steps=500 \
    trainer.total_epochs=10 "$@"
