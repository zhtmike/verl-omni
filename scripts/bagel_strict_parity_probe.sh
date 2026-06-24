#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${BAGEL_PARITY_DUMP_DIR:-}" ]]; then
  export BAGEL_PARITY_DUMP_DIR="$(pwd)/bagel_parity_dumps/verl"
  rm -rf "$BAGEL_PARITY_DUMP_DIR"
fi
mkdir -p "$BAGEL_PARITY_DUMP_DIR"

# Strict parity probe:
# - rollout.n=2 gives nonzero per-prompt advantages.
# - bypass_mode=True uses rollout log-probs as the PPO old-policy anchor,
#   matching official flow_grpo's in-process BAGEL setup.
# - rollout_is/rollout_rs stay disabled, so no extra verl-omni IS/RS weights
#   are applied to the actor objective.
# - LoRA dtype is aligned with official bf16.
bash scripts/bagel_mini_train_probe.sh \
  actor_rollout_ref.rollout.n=2 \
  algorithm.rollout_correction.bypass_mode=True \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  actor_rollout_ref.model.lora_dtype=bfloat16 \
  actor_rollout_ref.rollout.seed=0 \
  actor_rollout_ref.rollout.algo.sde_window_seed=0 \
  "$@"
