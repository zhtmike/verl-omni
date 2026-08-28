# verl_omni/pipelines/qwen_image_edit_flow_grpo/

## Responsibility
Flow-GRPO pipeline for **Qwen-Image-Edit-Plus** (`QwenImageEditPlusPipeline` architecture, `flow_grpo` algorithm) — image-conditioned editing (I2I) rather than t2i. Reuses Qwen-Image's T2I logic via multiple inheritance and adds the condition-latent injection hooks from `DiffusionI2IModelBase`. Drives `trainer/main_diffusion.py` (typically via `recipe/flowgrpo_trainer/`); dataset supplies source images as condition inputs parsed by `ImageGenerationRequest`.

## Design
- `diffusers_training_adapter.py`: `@DiffusionModelBase.register("QwenImageEditPlusPipeline", algorithm="flow_grpo")` class `QwenImageEditPlusFlowGRPO(DiffusionI2IModelBase, QwenImage)` — inherits `QwenImage`'s scheduler/input/sampling logic; overrides:
  - `prepare_processor_files`: writes the missing `processor/config.json` (`{"model_type": "qwen2_vl"}`) that Qwen-Image-Edit checkpoints omit, returning the processor dir.
  - `prepare_condition`: extracts condition image latents from the micro-batch.
  - `inject_condition`: concat-crop injection onto `hidden_states` with `_target_seq_len` slicing (base `DiffusionI2IModelBase` pattern).
- `vllm_omni_rollout_adapter.py`: `@VllmOmniPipelineBase.register("QwenImageEditPlusPipeline", algorithm="flow_grpo")` class `QwenImageEditPlusPipelineWithLogProb(QwenImageTokenIdPromptMixin, QwenImageEditPlusPipeline)` — swaps the upstream scheduler for `FlowMatchSDEDiscreteScheduler.from_pretrained`, encodes prompts via the Qwen mixin, and validates condition-image sizing (square images required unless pipeline height/width set, so condition latent lengths stay constant per batch).
- `common.py`: edit-specific helpers for condition image → latent preprocessing shared by rollout and training sides.

## Flow
1. Rollout: request carries source image(s) (`ImageGenerationRequest` candidate keys); `QwenImageEditPlusPipelineWithLogProb` encodes prompt + condition image, denoises with the SDE scheduler recording trajectory log-probs.
2. Training: dispatcher (`utils.prepare_model_inputs`) calls `QwenImage.prepare_model_inputs` (inherited), then, seeing an `DiffusionI2IModelBase` subclass, calls `prepare_condition` → `inject_condition` to concat condition latents onto `hidden_states`.
3. `forward_and_sample_previous_step` (inherited from `QwenImage`) runs the transformer (prediction sliced back to `_target_seq_len` by `DiffusionI2IModelBase.forward`) and calls `FlowMatchSDEDiscreteScheduler.sample_previous_step` for `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)`; rewards score edited output vs. source images via the reward loop.

## Integration
- Consumed by: diffusion trainer/engine via `(QwenImageEditPlusPipeline, flow_grpo)` registry keys.
- Depends on: `pipelines.qwen_image_flow_grpo` (`QwenImage` adapter, `QwenImageTokenIdPromptMixin`), `pipelines.model_base.DiffusionI2IModelBase`, `pipelines.schedulers`.
- Key entry points: `QwenImageEditPlusFlowGRPO`, `QwenImageEditPlusPipelineWithLogProb`.
- Sibling differences: the I2I counterpart of `qwen_image_flow_grpo` — same architecture family and algorithm, distinguished solely by condition-image injection and processor-config repair.
