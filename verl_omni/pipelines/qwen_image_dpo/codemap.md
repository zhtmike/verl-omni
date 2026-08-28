# verl_omni/pipelines/qwen_image_dpo/

## Responsibility
Wires Qwen-Image into the diffusion DPO algorithm (`algorithm="dpo"`): online rollout pairs scored by a preference signal, trained with the pairwise DPO loss (`DPOLoss`, `@register_diffusion_loss("dpo")` in `verl_omni/trainer/diffusion/diffusion_algos.py`). Provides both the training-side diffusers adapter and the rollout-side vLLM-Omni pipeline.

## Design
- Registration: `QwenImageDPO` via `@DiffusionModelBase.register("QwenImagePipeline", algorithm="dpo")` (training); `QwenImageDPOPipeline` via `@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dpo")` (rollout). Note it registers the same `"QwenImagePipeline"` architecture as flow_grpo/NFT but with `algorithm="dpo"`.
- Training adapter (`diffusers_training_adapter.py`) is self-contained (does not subclass the flow_grpo `QwenImage`): `build_scheduler` loads `FlowMatchEulerDiscreteScheduler.from_pretrained(..., subfolder="scheduler")`, `build_transformer_inputs` assembles `QwenImageTransformer2DModel` kwargs, and `_apply_true_cfg` implements norm-preserving CFG.
- Rollout adapter subclasses vLLM-Omni's `QwenImagePipeline` directly and implements the step-execution protocol: `prepare_encode` (token-id prompt extraction via `_extract_step_prompt_ids`, prompt embeds, latents, per-request scheduler), `denoise_step` (CFG-aware noise prediction kept in FP32), `step_scheduler` (FP32 scheduler step), `post_decode` (VAE decode + `with_rollout_data` attaching `latents_clean` and prompt embeddings).
- `encode_prompt`/`_get_qwen_prompt_embeds` re-implement token-id-native text encoding (last hidden state, template-prefix trimming via `prompt_template_encode_start_idx`, padding to max length).
- Unlike the NFT sibling, the pairwise preference signal comes from outside: pairs are formed by `DPOLoss.build_online_dpo_adjacent_pairs`/`build_online_dpo_pair_indices` (adjacent samples sharing a prompt uid), not by a reward-prob NFT objective.

## Flow
1. Entry: `python3 -m verl_omni.trainer.main_diffusion` with `algorithm.trainer_type=direct_preference` and `loss_mode=dpo` (e.g. `examples/dpo_trainer/qwen_image/run_qwen_image_online_dpo_lora.sh`).
2. Online rollout: engine calls `prepare_encode` → repeated `denoise_step`/`step_scheduler` → `post_decode`; each request returns decoded images plus `rl={"latents_clean": ...}` and positive/negative `prompt_embeds` (+masks) via `with_rollout_data` (CPU).
3. Preference wiring: multiple outputs per prompt are grouped into chosen/rejected pairs; the DPO loss computes per-pair flow-matching log-probabilities under the policy and reference models and optimizes `-logsigmoid(beta * Δ)`.
4. Training: `QwenImageDPO.prepare_model_inputs` builds transformer kwargs for policy (and negative-prompt CFG inputs when `true_cfg_scale > 1`); `forward_and_sample_previous_step` runs the transformer forward (with optional `_apply_true_cfg`) used for log-prob evaluation.

## Integration
- Consumed by: `verl_omni/pipelines/__init__.py`; `main_diffusion.py` → `DirectPreferenceRayTrainer`.
- Depends on: `verl_omni.pipelines.model_base`, `verl_omni.pipelines.diffusion_rollout_output` (`rollout_output`, `with_rollout_data`), `qwen_image_flow_grpo.common` (`apply_true_cfg`, `build_img_shapes`), `verl_omni.trainer.diffusion.diffusion_algos.DPOLoss`, vLLM-Omni `QwenImagePipeline`.
- Key entry points: `QwenImageDPO`, `QwenImageDPOPipeline`, `build_transformer_inputs`, `denoise_step`/`step_scheduler`/`post_decode`.
- Differences from the NFT sibling: pairwise DPO loss with reference model instead of NFT reward-prob weighting; emits only `latents_clean` (no `train_timesteps` from rollout).
