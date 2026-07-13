#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA training with omni V1 trainer.
# Hardware: 4× H100 80GB.
#
# Uses the omni_trainer.yaml config (inherits ppo_trainer + omni_model) and
# overrides all recipe-specific fields via CLI. No separate recipe YAML needed.
# Adapter registration (OmniModelBase, OmniRolloutPipelineBase) and
# processor/tokenizer loading are handled automatically — no monkey-patches.
set -x

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Make verl_omni available to Ray workers (no monkey-patch modules needed).
export VERL_USE_EXTERNAL_MODULES=verl_omni

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/math/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/math/test.parquet"}

python3 -m verl_omni.trainer.main_omni \
    --config-name=omni_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.truncation=left \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.architecture=Qwen3OmniMoeForConditionalGeneration \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar" \
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode=thinker_only" \
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.model_type=qwen3_omni_moe" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=100000000 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward.reward_manager.name=dapo \
    trainer.val_before_train=false \
    trainer.project_name=qwen3_omni_thinker_rl \
    trainer.experiment_name=gspo_lora_math \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=25 \
    trainer.total_epochs=5 \
    "$@"
