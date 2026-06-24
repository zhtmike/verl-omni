#!/usr/bin/env bash
set -euo pipefail

CONDA_SH=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}
CONDA_ENV=${CONDA_ENV:-flow_grpo}
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi
: "${CONDA_PREFIX:?Set CONDA_SH/CONDA_ENV or activate the probe conda environment before running.}"

export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTHONNOUSERSITE=1

# Torch in this env can otherwise pick up ~/.local's older NCCL before the
# env-local nvidia-nccl-cu12 package, causing `ncclCommShrink` import failures.
export LD_PRELOAD="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"

PROBE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "${BAGEL_PARITY_DUMP_DIR:-}" ]]; then
  export BAGEL_PARITY_DUMP_DIR="${PROBE_ROOT}/bagel_parity_dumps/official"
  rm -rf "$BAGEL_PARITY_DUMP_DIR"
fi
mkdir -p "$BAGEL_PARITY_DUMP_DIR"

USE_FLASH_ATTN_SHIM=${USE_FLASH_ATTN_SHIM:-0}
if [[ "$USE_FLASH_ATTN_SHIM" == "1" ]]; then
  export PYTHONPATH="${PROBE_ROOT}/scripts/flash_attn_compat:${PYTHONPATH:-}"
fi

FLOW_GRPO_ROOT=${FLOW_GRPO_ROOT:-"$HOME/gitlocal/flow_grpo"}
PICKSCORE_DATA=${PICKSCORE_DATA:-"$HOME/data/pickscore"}
BAGEL_MODEL=${BAGEL_MODEL:-"$HOME/models/ByteDance-Seed/BAGEL-7B-MoT"}
PROBE_RESOLUTION=${PROBE_RESOLUTION:-512}
FSDP_CONFIG=${FSDP_CONFIG:-${PROBE_ROOT}/scripts/accelerate_fsdp_cpu_offload.yaml}

cd "$FLOW_GRPO_ROOT"

# Keep this to one epoch, one sampling batch, and two samples per prompt.
# With 4 processes and per-rank sample batch size 1, this produces 4 images
# grouped as 2 samples per prompt across ranks.
timeout 10m accelerate launch \
  --config_file "$FSDP_CONFIG" \
  --num_processes=4 \
  --main_process_port 29509 \
  scripts/train_bagel.py \
  --config config/grpo.py:pickscore_bagel_lora \
  --config.debug=True \
  --config.num_epochs=1 \
  --config.dataset="$PICKSCORE_DATA" \
  --config.pretrained.model="$BAGEL_MODEL" \
  --config.resolution="$PROBE_RESOLUTION" \
  --config.sample.train_batch_size=1 \
  --config.sample.num_image_per_prompt=2 \
  --config.sample.num_batches_per_epoch=1 \
  --config.train.batch_size=1 \
  --config.train.gradient_accumulation_steps=1 \
  --config.save_freq=999 \
  --config.eval_freq=999 \
  --config.run_name=bagel_official_one_step_probe \
  --config.logdir=logs/debug \
  "$@"
