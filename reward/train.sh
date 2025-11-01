#!/bin/bash
cd /lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/RL/verl
pip install -e .

export RAY_num_server_call_thread=1
export RAY_TMPDIR="${OUTPUT_DIR}/ray_tmp"
mkdir -p ${RAY_TMPDIR}
ray start --head --port=7801 --dashboard-port=7802 --num-cpus=32

OUTPUT_DIR="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/output_models/qwen2_5-7b-instruct-40k-yitu"
MODEL_NAME="Qwen2.5-7B-Instruct"
DATA_NAME="qwen25_7b_ins-40k-yitu"
MODEL_PATH="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/models/qwen2_5-7b-instruct"
DATA_TRAIN_PATH="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/data/40k_dapo_yitu/train_parquet/train.parquet"
DATA_VAL_PATH="/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/train/data/40k_dapo_yitu/val_parquet/val.parquet"
mkdir -p ${OUTPUT_DIR}

#=============================


# Model configuration
ROLLOUT_LOG_DIR="${OUTPUT_DIR}/rollout_log/${EXPERIMENT_NAME}"
VALIDATION_LOG_DIR="${OUTPUT_DIR}/validation_log/${EXPERIMENT_NAME}"
# Experiment configuration
EXPERIMENT_NAME="${MODEL_NAME}_${DATA_NAME}"


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

export WANDB_MODE=offline
set -x

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_TRAIN_PATH} \
    data.val_files=${DATA_VAL_PATH} \
    data.train_batch_size=64 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
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
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.rollout_data_dir="${ROLLOUT_LOG_DIR}" \
    trainer.validation_data_dir="${VALIDATION_LOG_DIR}" \
    trainer.total_training_steps=500 \
    trainer.total_epochs=10 $@