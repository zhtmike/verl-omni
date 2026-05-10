# DanceGRPO trainer examples

This directory contains launch scripts for training diffusion models with the
**DanceGRPO** algorithm
([paper](https://arxiv.org/abs/2505.07818),
 [reference implementation](https://github.com/ByteDance-Seed/DanceGRPO)).

DanceGRPO is structurally a sibling of FlowGRPO:

| Component                | FlowGRPO                            | DanceGRPO                          |
|--------------------------|-------------------------------------|------------------------------------|
| Advantage estimator      | Group-normalised outcome advantage  | **Same**                           |
| Policy loss              | Clipped-PPO                         | **Same**                           |
| SDE step formula         | `std_dev_t = sqrt(σ/(1-σ))·η`       | `std_dev_t = η·sqrt(Δt)`           |
| SDE window               | Optional contiguous window of steps | All steps inject noise (no window) |

The shared `FlowMatchSDEDiscreteScheduler` exposes both formulas via
`sde_type="sde"` (FlowGRPO) and `sde_type="dance"` (DanceGRPO).

## Algorithm dispatch

VeRL-Omni resolves the per-algorithm adapter pair by composing two registries
keyed on `(architecture, algorithm)`:

* `DiffusionModelBase` — training-side adapter
  (`verl_omni/pipelines/qwen_image_dance_grpo/diffusers_training_adapter.py`).
* `VllmOmniPipelineBase` — rollout-side pipeline
  (`verl_omni/pipelines/qwen_image_dance_grpo/vllm_omni_rollout_adapter.py`).

The active algorithm is read from `actor_rollout_ref.model.algorithm`, which
defaults to `${oc.select:algorithm.adv_estimator,flow_grpo}` — i.e. setting
`algorithm.adv_estimator=dance_grpo` is enough to pick the DanceGRPO adapters,
loss, and advantage estimator together.

## Scripts

* [`run_qwen_image_ocr_lora.sh`](run_qwen_image_ocr_lora.sh) — Qwen-Image LoRA
  fine-tune on the OCR reward, mirroring the FlowGRPO example one-to-one.
