# verl_omni/pipelines/boogu_image_flow_grpo/

## Responsibility
Flow-GRPO pipeline for the **Boogu-Image** model (`BooguImagePipeline` architecture, `flow_grpo` algorithm), supporting both T2I and Edit/TI2I modes. Structurally a close sibling of `qwen_image_flow_grpo` — a diffusers-style transformer pipeline with an SDE scheduler — but with Boogu-specific scheduler constants, RoPE axes, and text-CFG. Drives `trainer/main_diffusion.py` (typically via `recipe/flowgrpo_trainer/`).

## Design
- `diffusers_training_adapter.py`: `@DiffusionModelBase.register("BooguImagePipeline", algorithm="flow_grpo")` class `BooguImage(DiffusionModelBase)`. Distinct overrides: `prepare_processor_files` (fixes processor config before HF load), `build_module` (loads the canonical class because the checkpoint has no `auto_map`), `_make_config_checkpointable` (config wrapper making `save_pretrained` a no-op for FSDP checkpointing). Scheduler built on the SDE scheduler with Boogu config (`_load_vae_scale_factor`, `_scheduler_num_train_timesteps`).
- `common.py`: `build_boogu_native_scheduler`, `configure_boogu_sde_timesteps` (registers `use_dynamic_shifting=False, shift=1.0`), `boogu_timestep_from_scheduler` (timestep rescaling), `apply_boogu_text_cfg`, `get_boogu_freqs_cis` (RoPE axes for Boogu), `resolve_text_guidance_scale`.
- `vllm_omni_rollout_adapter.py`: `@VllmOmniPipelineBase.register("BooguImagePipeline", algorithm="flow_grpo")` class `BooguImagePipelineWithLogProb(QwenImageTokenIdPromptMixin, BooguImagePipeline)` — reuses Qwen-Image prompt-id encoding mixin on top of the vllm-omni `BooguImagePipeline`; supports both t2i and edit (condition-image) requests; `_collate_prompt_batch` uses `pipelines/request_batch.py` helpers (`collate_prompt_rows`, `sample_per_sample_sde_windows`); `get_rollout_post_process_func` wraps outputs with `wrap_rollout_postprocessor`.

## Flow
1. Rollout: `BooguImagePipelineWithLogProb.diffuse(...)` encodes prompts (token ids → `_get_boogu_prompt_embeds` via the Qwen mixin), runs SDE denoising with per-sample windows, records latents/timesteps/log-probs into `diffusion_rollout_output.rollout_output`.
2. Training: `BooguImage.build_scheduler`/`set_timesteps` configures the SDE scheduler from Boogu's checkpoint config; `prepare_model_inputs` builds Boogu transformer inputs (RoPE freqs from `get_boogu_freqs_cis`, timestep rescaling via `boogu_timestep_from_scheduler`); `forward` override handles Boogu's output conversion.
3. `forward_and_sample_previous_step` applies text CFG (`apply_boogu_text_cfg`) then calls `FlowMatchSDEDiscreteScheduler.sample_previous_step` for `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)` feeding the flow-GRPO loss; rewards via the standard reward loop.

## Integration
- Consumed by: diffusion trainer/engine via the `(BooguImagePipeline, flow_grpo)` registry keys.
- Depends on: `pipelines/schedulers`, `pipelines.qwen_image_flow_grpo.common` (prompt mixin, `coalesce_not_none`), `pipelines.request_batch`, `pipelines.diffusion_rollout_output`, vllm-omni `BooguImagePipeline`.
- Key entry points: `BooguImage`, `BooguImagePipelineWithLogProb`.
- Sibling differences: extends `QwenImageTokenIdPromptMixin` rather than re-implementing encoding; carries Boogu-native scheduler/RoPE handling and processor-file preparation absent from Qwen-Image pipelines.
