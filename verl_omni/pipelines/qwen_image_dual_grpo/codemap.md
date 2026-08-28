# verl_omni/pipelines/qwen_image_dual_grpo/

## Responsibility
Dual-policy GRPO pipeline for **Qwen-Image** (`QwenImagePipeline` architecture, `dual_grpo` algorithm). Optimizes **two** policies jointly: the Qwen2.5-VL text encoder (treated as an AR policy that rewrites the prompt) and the DiT diffusion policy. Rollout-only directory — it subclasses the base flow-GRPO rollout pipeline and reuses the existing `QwenImage` training adapter from `qwen_image_flow_grpo`. Drives `trainer/main_diffusion.py` (typically via `recipe/flowgrpo_trainer/` with `algorithm=dual_grpo`).

## Design
- `vllm_omni_rollout_adapter.py`: `@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dual_grpo")` class `QwenImagePipelineWithDualLogProb(QwenImagePipelineWithLogProb)` — the only registration in this directory (no `diffusers_training_adapter.py`; `DiffusionModelBase` lookup still resolves `("QwenImagePipeline", ...)` per-algorithm via the base flow-GRPO adapter).
- Two-path rollout:
  1. Text path — `generate_text_encoder_response` / `_get_qwen_text_response` autoregressively samples one rewritten-prompt response per request from the Qwen2.5-VL text encoder, captured in the `TextEncoderGenerationResult` dataclass (`llm_response_ids`, `llm_log_probs`, `text_encoder_responses`).
  2. Diffusion path — inherited `QwenImagePipelineWithLogProb` SDE denoising producing DiT trajectory log-probs.
- Imports `pipelines/request_batch.py` helpers (`collate_prompt_rows`, `collate_prompt_mask`, `sample_per_sample_sde_windows`, `split_diffusion_output_by_request`) and `qwen_image_flow_grpo.common.build_img_shapes` / `coalesce_not_none`.

## Flow
1. Request batch arrives via `DiffusionRequestBatch`; per request the pipeline first samples text-encoder tokens and their log-probs (`llm_all_log_probs`), then runs SDE image diffusion conditioned on the generated text, recording `all_log_probs`.
2. Output splits per request via `split_diffusion_output_by_request`; both log-prob sets plus `text_encoder_responses` go into `pipelines.diffusion_rollout_output.rollout_output`.
3. Training: the trainer computes a combined objective — GRPO ratios for the AR text tokens (encoder policy) and the standard flow-GRPO SDE ratio for DiT steps (`QwenImage.forward_and_sample_previous_step` reused unchanged); rewards on the final images update both policies.

## Integration
- Consumed by: diffusion trainer/engine via the `(QwenImagePipeline, dual_grpo)` rollout key; training side resolved by the `("QwenImagePipeline", "dual_grpo")`→ shared adapter if registered externally, else via `external_lib` fallback.
- Depends on: `pipelines.qwen_image_flow_grpo` (rollout base class + common), `pipelines.request_batch`, `pipelines.diffusion_rollout_output`.
- Key entry points: `QwenImagePipelineWithDualLogProb`, `TextEncoderGenerationResult`.
- Sibling differences: only Qwen-Image GRPO variant with a second (text-encoder) policy; no new training adapter — pure rollout-layer specialization of `qwen_image_flow_grpo`.
