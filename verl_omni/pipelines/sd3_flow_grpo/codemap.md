# verl_omni/pipelines/sd3_flow_grpo/

## Responsibility
Flow-GRPO pipeline for **Stable Diffusion 3 / 3.5** (`StableDiffusion3Pipeline` architecture, `flow_grpo` algorithm), t2i with dual CLIP+T5 text encoders. Drives `trainer/main_diffusion.py` (typically via `recipe/flowgrpo_trainer/`); rewards scored on generated images via the standard reward loop.

## Design
- `diffusers_training_adapter.py`: `@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="flow_grpo")` class `StableDiffusion3FlowGRPO(DiffusionModelBase)` — `FlowMatchSDEDiscreteScheduler` with SD3-style resolution shift (`_calculate_shift`, `_sd3_image_seq_len`, local reimplementation of diffusers' `calculate_shift`); `build_transformer_inputs` plus the standard `prepare_model_inputs` / `forward_and_sample_previous_step`.
- `common.py`: `SD3TokenIdPromptMixin` with `_get_clip_prompt_embeds_from_ids`, `_get_t5_prompt_embeds_from_ids`, `encode_prompt_from_token_ids` — encodes per-text-encoder token ids (produced once by the agent loop via `actor_rollout_ref.model.extra_tokenizers`) so no decode/re-encode happens at rollout; `pad_token_id_batch`.
- `vllm_omni_rollout_adapter.py`: `@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="flow_grpo")` class `StableDiffusion3PipelineWithLogProb(SD3TokenIdPromptMixin, StableDiffusion3Pipeline)`; `supports_request_batch = True`. Replaces the Euler scheduler with the SDE scheduler; `forward` collects `all_latents`/`all_log_probs`/`all_timesteps` and ships prompt embeddings (sequence + pooled) through `trajectory_*` metadata. Also installs a module-level post-process override: `pipeline_sd3.get_sd3_image_post_process_func = get_latent_post_process_func` (vLLM-Omni resolves the factory before pipeline init).
- CFG: plain SD3 `guidance_scale`; `guidance_scale <= 1` skips the negative branch entirely (halves transformer NFE).

## Flow
1. Agent loop tokenizes prompts once per text encoder; request batch → `StableDiffusion3PipelineWithLogProb.forward` encodes embeddings from ids via the mixin, denoises with SDE-window sampling, records trajectory + embeddings into rollout output.
2. Training: `StableDiffusion3FlowGRPO.set_timesteps` mirrors the rollout schedule (shift from image seq len); `prepare_model_inputs` slices trajectory latents/timesteps per step; `forward_and_sample_previous_step` runs the transformer (CFG when enabled) and calls `FlowMatchSDEDiscreteScheduler.sample_previous_step` for `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)`.

## Integration
- Consumed by: diffusion trainer/engine via `(StableDiffusion3Pipeline, flow_grpo)` registry keys.
- Depends on: `pipelines/schedulers`, `pipelines.model_base`, vllm-omni `pipeline_sd3`.
- Key entry points: `StableDiffusion3FlowGRPO`, `StableDiffusion3PipelineWithLogProb`, `SD3TokenIdPromptMixin`.
- Sibling differences: dual-encoder (CLIP+T5) token-id prompt path and pooled-embedding transport; module-level post-process factory override; sibling `sd3_dpo` shares the architecture but trains a forward-process DPO objective and ships no rollout adapter.
