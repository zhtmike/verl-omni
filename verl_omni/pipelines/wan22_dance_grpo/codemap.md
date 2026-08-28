# verl_omni/pipelines/wan22_dance_grpo/

## Responsibility
DanceGRPO pipeline for **Wan2.2 video** (`WanPipeline` architecture, `dance_grpo` algorithm), supporting text-to-video (T2V) and image-to-video (I2V) with both single-transformer (TI2V-5B) and dual-transformer MoE (A14B) checkpoints. Drives `trainer/main_diffusion.py` (typically via `recipe/dancegrpo_trainer/`); rewards scored on generated video via the standard reward loop.

## Design
- `diffusers_training_adapter.py`: `@DiffusionModelBase.register("WanPipeline", algorithm="dance_grpo")` class `Wan22DanceGRPO(DiffusionModelBase)` — `FlowMatchSDEDiscreteScheduler` with SD3-style time shift (`_configure_wan_scheduler`, `common.sd3_time_shift`); `prepare_model_inputs` builds `WanTransformer3DModel` inputs including `encoder_hidden_states_image` (image conditioning for I2V); `forward_and_sample_previous_step` applies CFG via `common.apply_cfg` when `true_cfg_scale > 1.0` then calls `scheduler.sample_previous_step(..., sde_type=model_config.algo.sde_type, return_sqrt_dt=True)`. Overrides `preserve_fp32_modules()` → False (mixed-precision FSDP units).
- `common.py`: `sd3_time_shift` (sigma shifting), `apply_cfg`, `flatten`, `seed_from_prompt_ids` (deterministic per-prompt seeding for rollout noise).
- `vllm_omni_rollout_adapter.py`: `@VllmOmniPipelineBase.register("WanPipeline", algorithm="dance_grpo")` class `Wan22DanceGRPOPipelineWithLogProb(Wan22Pipeline)` (from `vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2`) — replaces the UniPC/Euler scheduler with the SDE scheduler, enables VAE slicing, and captures per-step log-prob trajectories; supports `expand_timesteps` mode for I2V conditioning. The `__init__.py` wraps this import in try/except with an `_UnavailableModule` stub raising "Wan22 Dance GRPO requires GPU (CUDA/NPU)" on CPU-only machines.

## Flow
1. Rollout: `Wan22DanceGRPOPipelineWithLogProb` denoises video latents (5D tensors: batch × channels × frames × H × W) with SDE noise, recording `all_latents`/`all_timesteps`/log-probs; I2V requests condition via image latents with expand-timesteps scheduling.
2. Training: `Wan22DanceGRPO.set_timesteps` mirrors the shifted sigma schedule; `prepare_model_inputs` slices `latents[:, step]` / `timesteps[:, step]` and builds positive/negative input dicts.
3. `forward_and_sample_previous_step` forwards `WanTransformer3DModel`, applies `apply_cfg`, and gets `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)` from `FlowMatchSDEDiscreteScheduler.sample_previous_step` (typically `sde_type="dance_sde"` for DanceGRPO); policy gradient computed in `trainer/diffusion/diffusion_algos.py`.

## Integration
- Consumed by: diffusion trainer/engine via `(WanPipeline, dance_grpo)` registry keys.
- Depends on: `pipelines/schedulers`, `pipelines.model_base`, diffusers `WanTransformer3DModel`, vllm-omni `Wan22Pipeline`.
- Key entry points: `Wan22DanceGRPO`, `Wan22DanceGRPOPipelineWithLogProb`.
- Sibling differences: only video pipeline in the GRPO family and the only `dance_grpo` registration; 5D video latents, image conditioning via `encoder_hidden_states_image`, dual-transformer MoE support, and GPU-required rollout import guard.
