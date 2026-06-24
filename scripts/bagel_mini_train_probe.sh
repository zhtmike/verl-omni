#!/usr/bin/env bash
set -euo pipefail

CONDA_SH=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}
CONDA_ENV=${CONDA_ENV:-verl-omni-022}
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi
: "${CONDA_PREFIX:?Set CONDA_SH/CONDA_ENV or activate the probe conda environment before running.}"

export LD_LIBRARY_PATH=${CONDA_PREFIX}/cuda-compat:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export VLLM_USE_DEEP_GEMM=0

cd "$(dirname "$0")/.."

ray stop --force >/dev/null 2>&1 || true

timeout 10m bash examples/flowgrpo_trainer/run_bagel_flowgrpo_lora.sh \
  data.train_batch_size=4 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.logger='["console"]' \
  trainer.project_name=flow_grpo_debug \
  trainer.experiment_name=bagel_mini_probe \
  trainer.resume_mode=disable \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=0 \
  trainer.test_freq=0 \
  trainer.log_val_generations=0 \
  "$@"
