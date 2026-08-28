# verl_omni/pipelines/schedulers/

## Responsibility
Custom diffusion schedulers shared by all GRPO-family pipelines. Provides `FlowMatchSDEDiscreteScheduler`, an SDE extension of diffusers' `FlowMatchEulerDiscreteScheduler` that supports stochastic reverse steps with per-step Gaussian log-probabilities — the mathematical core of flow-GRPO, DanceGRPO, MixGRPO, and GRPO-Guard training. Based on the FlowGRPO paper (arXiv:2505.05470) and the flow_grpo diffusers patch.

## Design
- `FlowMatchSDEDiscreteScheduler(FlowMatchEulerDiscreteScheduler)` — overrides `step` and adds `sample_previous_step`; both support three reverse-step formulations selected by `sde_type`:
  - `"sde"`: FlowGRPO SDE — `std_dev_t = sqrt(sigma / (1-sigma)) * noise_level`; mean combines drift terms; log-prob includes optional Gaussian normalizer (`include_logprob_normalizer`).
  - `"cps"`: cosine-preserving step using `pred_original_sample` and `noise_estimate` with `std_dev_t = sigma_prev * sin(noise_level * pi/2)`.
  - `"dance_sde"`: DanceGRPO score-based SDE correction with ODE mean `sample + dsigma * model_output` plus `-0.5 * eta^2 * score_estimate * dsigma`, stable near sigma≈1.
- `FlowMatchSDEDiscreteSchedulerOutput(BaseOutput)` dataclass carries `(prev_sample, log_prob, prev_sample_mean, std_dev_t)`.
- Key knobs: `noise_level` (stochasticity η, from `DiffusionModelConfig.algo.noise_level`), `prev_sample` (teacher-forced sample: compute log-prob of an existing rollout sample instead of re-sampling), `return_sqrt_dt` (GRPO-Guard importance-ratio normalization), `per_token_timesteps` (not yet implemented).

## Flow
1. A pipeline's training adapter (e.g. `QwenImage.build_scheduler`) loads the scheduler via `from_pretrained(model_path, subfolder="scheduler")` and calls `set_timesteps` with model-specific sigmas/shift.
2. Rollout: `step(...)` runs one reverse step with fresh `variance_noise` (`randn_tensor`), emitting `FlowMatchSDEDiscreteSchedulerOutput` whose `log_prob` is stored per denoising step.
3. Training: `sample_previous_step(sample=latents[:, step], model_output=noise_pred, timestep=timesteps[:, step], prev_sample=latents[:, step+1], sde_type=..., return_logprobs=True)` recomputes old/new log-probs against the recorded trajectory; log_prob is mean-reduced over all non-batch dims; `return_sqrt_dt=True` additionally yields `sqrt(-dt)` per batch element.

## Integration
- Consumed by: every flow/dance/mix-GRPO pipeline's `diffusers_training_adapter.py` (qwen_image_flow_grpo, qwen_image_edit_flow_grpo, qwen_image_dual_grpo, sd3_flow_grpo, wan22_dance_grpo, bagel_flow_grpo, boogu_image_flow_grpo, ltx2_flow_grpo); DPO/NFT pipelines use the base diffusers schedulers instead.
- Depends on: `diffusers.FlowMatchEulerDiscreteScheduler`, `diffusers.utils.torch_utils.randn_tensor`.
- Key entry points: `FlowMatchSDEDiscreteScheduler.step`, `FlowMatchSDEDiscreteScheduler.sample_previous_step`; re-exported via `schedulers/__init__.py`.
