# verl_omni/pipelines/qwen_image_flow_grpo/

## Responsibility
Reference t2i flow-GRPO pipeline for **Qwen-Image** (`QwenImagePipeline` architecture, `flow_grpo` algorithm). Provides the training adapter (`QwenImage`), the log-prob-instrumented rollout pipeline (`QwenImagePipelineWithLogProb`), and Qwen-Image-specific math (True-CFG, img_shapes, prompt encoding). Drives the diffusion trainer path (`trainer/main_diffusion.py` → `trainer/diffusion/`), typically launched from `recipe/flowgrpo_trainer/`.

## Design
- `diffusers_training_adapter.py`: `@DiffusionModelBase.register("QwenImagePipeline", algorithm="flow_grpo")` class `QwenImage`. Scheduler is `FlowMatchSDEDiscreteScheduler` (from `pipelines/schedulers`), configured with resolution-dependent shift via diffusers' `calculate_shift` and linear sigmas (`_configure_qwen_image_scheduler`, `QWEN_IMAGE_VAE_SCALE_FACTOR = 8`).
- `common.py`: `apply_true_cfg` (norm-preserving classifier-free guidance combining conditional/negative predictions), `build_img_shapes` (per-sample `[(1, latent_h, latent_w)]`), `QwenImageTokenIdPromptMixin` (`_get_qwen_prompt_embeds` / `encode_prompt`) that encodes pre-tokenized prompt ids and unpads Qwen's vision-start prefix tokens.
- `vllm_omni_rollout_adapter.py`: `@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="flow_grpo")` class `QwenImagePipelineWithLogProb` — a diffusers pipeline subclass that runs denoising with SDE noise injection, records per-step latents/timesteps and log-probs, and returns them as rollout output consumed by the trainer.

## Flow
1. Rollout: async server instantiates `QwenImagePipelineWithLogProb` (resolved via `VllmOmniPipelineBase.get_pipeline_path`); requests parsed with `ImageGenerationRequest`; prompt encoded through `QwenImageTokenIdPromptMixin.encode_prompt`; denoise loop stores `all_latents`, `all_timesteps`, per-step log-probs into the rollout output.
2. Training step: engine (`workers/engine/fsdp/diffusers_impl.py`) builds the scheduler once via `QwenImage.build_scheduler`, then per micro-batch/step calls `utils.prepare_model_inputs` → `QwenImage.prepare_model_inputs`, which slices `latents[:, step]` and `timesteps[:, step] / 1000.0` and builds positive/negative input dicts (guidance vector when `guidance_embeds`).
3. `QwenImage.forward_and_sample_previous_step` forwards the transformer, applies `apply_true_cfg` when `model_config.pipeline.true_cfg_scale > 1.0`, then calls `scheduler.sample_previous_step(..., sde_type=model_config.algo.sde_type, return_logprobs=True)` with `prev_sample=latents[:, step + 1]` to produce `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)` for the flow-GRPO importance ratio and SDE-based policy gradient in `trainer/diffusion/diffusion_algos.py`.
4. Rewards: generated images scored via the reward loop configured in the recipe (e.g. image-reward / HPS-type scorers); advantages per prompt group feed the GRPO loss.

## Integration
- Consumed by: diffusion trainer/engine via the `(QwenImagePipeline, flow_grpo)` registry keys.
- Depends on: `pipelines/schedulers.FlowMatchSDEDiscreteScheduler`, `pipelines.model_base`, `pipelines.utils`, diffusers `QwenImageTransformer2DModel`, Ulysses SP monkey-patch applied in `DiffusionModelBase.get_class` (`apply_qwen_image_ulysses_mask_fix`).
- Key entry points: `QwenImage`, `QwenImagePipelineWithLogProb`; dataset class selected by recipe config (prompt-only t2i dataset).
- Sibling differences: this is the canonical t2i flow-GRPO implementation — `qwen_image_mix_grpo` and `qwen_image_dual_grpo` reuse its components; `qwen_image_edit_flow_grpo` adds condition-image (I2I) handling on the same architecture.
