# verl_omni/pipelines/qwen_image_diffusion_nft/

## Responsibility
Wires Qwen-Image text-to-image into the DiffusionNFT forward-process algorithm (`algorithm="diffusion_nft"`). It reuses the Qwen-Image flow_grpo machinery for prompting/shapes and adds the thin NFT-specific pieces: clean-latent + train-timestep rollout output and a plain forward training adapter.

## Design
- Registration: `QwenImageDiffusionNFT` via `@DiffusionModelBase.register("QwenImagePipeline", algorithm="diffusion_nft")` (training); `QwenImageDiffusionNFTPipeline` via `@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="diffusion_nft")` (rollout).
- Training adapter subclasses `QwenImage` from `verl_omni/pipelines/qwen_image_flow_grpo/diffusers_training_adapter.py` and only overrides `prepare_model_inputs` (builds `img_shapes` via `build_img_shapes`, optional guidance embeds, optional negative-prompt inputs) and `forward` (single transformer call, optional True-CFG via `apply_true_cfg`).
- Rollout adapter mixes in `QwenImageTokenIdPromptMixin` (from `qwen_image_flow_grpo/common.py`) for token-id-native prompt encoding; `_extract_step_prompt_ids`/`_tokenize_step_prompt` accept `prompt_token_ids` or raw text with the model's `prompt_template_encode`.
- Two rollout execution paths: `forward()` (full denoising loop delegating to `super().diffuse(...)`) and the step-based `prepare_encode`/`post_decode` pair using `StepRequestState` (per-request deep-copied scheduler).
- Because DiffusionNFT trains from the final clean latent with a forward-process objective, no reverse-SDE trajectory or log-prob tensors are collected — only `latents_clean` and `train_timesteps`.

## Flow
1. Entry: `python3 -m verl_omni.trainer.main_diffusion` with `algorithm.sample_source=online`, `loss_mode=diffusion_nft` (e.g. `examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora.sh`).
2. Rollout: `forward()` or `prepare_encode` → `_prepare_token_id_generation_context` (encode prompts + negatives, `prepare_latents`, `prepare_timesteps`, `build_img_shapes`) → denoise via upstream `QwenImagePipeline.diffuse` → `_decode_latents` → `with_rollout_data`/`rollout_output` attaches `latents_clean`, `train_timesteps`, and positive/negative `prompt_embeds` (+masks), moved to CPU.
3. Reward: externally configured reward function (e.g. `verl_omni/utils/reward_score/genrm_ocr.py:compute_score` with a Qwen3-VL reward model) scores images; `DiffusionNFTLoss` (`trainer/diffusion/diffusion_algos.py`) maps group advantages to `reward_prob` and samples train timesteps.
4. Training: `QwenImageDiffusionNFT.prepare_model_inputs`/`forward` run the transformer (with optional `apply_true_cfg`) on noised `latents_clean` at sampled timesteps for the NFT loss.

## Integration
- Consumed by: `verl_omni/pipelines/__init__.py`; `main_diffusion.py` → `DirectPreferenceRayTrainer` (requires `loss_mode=diffusion_nft`).
- Depends on: `verl_omni.pipelines.model_base`, `verl_omni.pipelines.qwen_image_flow_grpo.{common,diffusers_training_adapter}` (`QwenImage`, `QwenImageTokenIdPromptMixin`, `build_img_shapes`, `apply_true_cfg`, `coalesce_not_none`), `verl_omni.pipelines.diffusion_rollout_output`, vLLM-Omni `QwenImagePipeline`.
- Key entry points: `QwenImageDiffusionNFT`, `QwenImageDiffusionNFTPipeline`, `_prepare_token_id_generation_context`.
- Differences from flow_grpo sibling (`qwen_image_flow_grpo`): no per-step logprob/old-logprob collection, no reverse-sampling; emits final clean latents plus the timestep pool for the NFT forward-process objective.
