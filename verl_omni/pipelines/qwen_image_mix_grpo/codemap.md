# verl_omni/pipelines/qwen_image_mix_grpo/

## Responsibility
MixGRPO pipeline for **Qwen-Image** (`QwenImagePipeline` architecture, `mix_grpo` algorithm). The thinnest GRPO pipeline: both files are subclasses of the flow-GRPO adapters, differing only in **SDE window placement strategy** — MixGRPO restricts stochastic sampling to a sliding/positioned window over the denoising trajectory rather than full random per-step noise. Drives `trainer/main_diffusion.py` (typically via `recipe/mixgrpo_trainer/`).

## Design
- `diffusers_training_adapter.py` (29 lines): `@DiffusionModelBase.register("QwenImagePipeline", algorithm="mix_grpo")` class `QwenImageMixGRPO(QwenImage)` — empty body; inherits all scheduler/input/sampling logic; the algorithmic difference lives in rollout windowing and trainer-side loss handling.
- `vllm_omni_rollout_adapter.py`: `@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="mix_grpo")` class `QwenImageMixGRPOPipelineWithLogProb(QwenImagePipelineWithLogProb)`; overrides `prepare_encode(state: StepRequestState, ...)` to pin the SDE window before step execution draws it. Two strategies (knobs on `DiffusionRolloutAlgoConfig`: `sample_strategy`, `iters_per_group`, `sde_window_seed`):
  - `random` — window start seeded by `sde_window_seed + global_steps` (identical across rollout ranks).
  - `progressive` — deterministic slide, advancing by `sde_window_size` every `iters_per_group` iterations, implemented by collapsing the random draw range to one value.
  - Trainer state (`global_steps`) is forwarded by the diffusion agent loop via `sampling_params.extra_args`.

## Flow
1. Agent loop passes `global_steps` through request `extra_args`; `prepare_encode` fixes the window start per strategy.
2. Rollout: inherited `QwenImagePipelineWithLogProb` denoising runs with noise injected only inside the positioned SDE window; trajectory latents/timesteps/log-probs recorded as in flow-GRPO.
3. Training: inherited `QwenImageMixGRPO` (=`QwenImage`) `forward_and_sample_previous_step` recomputes log-probs on the recorded trajectory; the MixGRPO loss (ODE/SDE mixing) is handled in `trainer/diffusion/diffusion_algos.py`; rewards via the standard reward loop.

## Integration
- Consumed by: diffusion trainer/engine via `(QwenImagePipeline, mix_grpo)` registry keys.
- Depends on: `pipelines.qwen_image_flow_grpo` (both adapters), `pipelines.model_base`, vllm-omni `StepRequestState`/`DiffusionRequestBatch`.
- Key entry points: `QwenImageMixGRPO`, `QwenImageMixGRPOPipelineWithLogProb`.
- Sibling differences: minimal delta vs. `qwen_image_flow_grpo` — no new model/scheduler code; only window-position control distinguishes it.
