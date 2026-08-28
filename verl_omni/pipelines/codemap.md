# verl_omni/pipelines/

## Responsibility
Houses all model+algorithm "pipelines": per-model training adapters and rollout adapters that plug a specific diffusion (or omni AR) model into the verl-omni RL trainer. Each subdirectory pairs one model architecture with one RL algorithm (e.g. flow-GRPO for Qwen-Image, DanceGRPO for Wan2.2) and registers itself into shared class registries consumed by the trainer and rollout engine. Shared plumbing lives at the root: `model_base.py` (registry ABCs), `utils.py` (dispatcher helpers), `request_batch.py`, `diffusion_rollout_output.py`, `non_diffusers_model_base.py`.

## Design
- **Dual registry pattern** (`model_base.py`): `DiffusionModelBase.register(architecture, algorithm)` registers *training* adapters keyed on `(architecture, algorithm)` where `architecture` is the `_class_name` from the model's `model_index.json` and `algorithm` is `DiffusionModelConfig.algorithm` (e.g. `"flow_grpo"`, `"dance_grpo"`, `"dpo"`, `"diffusion_nft"`, `"mix_grpo"`, `"dual_grpo"`). `VllmOmniPipelineBase.register(...)` registers *rollout* pipeline classes (custom diffusers pipeline subclasses with log-prob instrumentation); the async rollout server resolves them via `VllmOmniPipelineBase.get_pipeline_path()` (`workers/rollout/vllm_rollout/vllm_omni_async_server.py`).
- **Omni registries**: `OmniModelBase.register(architecture, stage)` for AR omni training adapters (stage = thinker/talker/all) and `OmniRolloutPipelineBase.register(model_type)` for vLLM-Omni pipeline topology (per-stage configs, rollout flags, HF overrides).
- **Dispatcher helpers** (`utils.py`): `prepare_model_inputs`, `build_scheduler`, `set_timesteps`, `forward`, `forward_and_sample_previous_step` all delegate through `DiffusionModelBase.get_class(model_config)` so trainer/engine code is architecture-agnostic. `DiffusionI2IModelBase` adds `prepare_condition`/`inject_condition` (concat-crop default) for image-conditioned models. DPO-pair helpers: `sample_noise_and_timesteps`, `prepare_noisy_latents`, `get_sigmas`; `ImageGenerationRequest.from_request_payload` parses rollout request payloads (prompt + condition images) for t2i and i2i backends.
- **Adapter file convention** per pipeline dir: `diffusers_training_adapter.py` (train side, subclasses `DiffusionModelBase`/`DiffusionI2IModelBase`), `vllm_omni_rollout_adapter.py` (rollout side, registered via `VllmOmniPipelineBase`), `common.py` (model-specific math/constants), `agent_loop.py` (optional per-model agent-loop config).
- `__init__.py` imports every subpackage so registrations fire on `import verl_omni.pipelines`; registration is import-time via class decorators.
- Trainer entry points: `trainer/main_diffusion.py` + `trainer/diffusion/` (uses `get_class_by_name`, engine impls in `workers/engine/fsdp/diffusers_impl.py`); `trainer/main_omni.py` + `trainer/omni/` for the omni path. Recipe scripts under `recipe/<algo>_trainer/` and configs under `scripts/` drive these mains.

## Flow
1. `import verl_omni.pipelines` executes each subpackage's `__init__.py`, firing `@DiffusionModelBase.register` / `@VllmOmniPipelineBase.register` decorators.
2. `main_diffusion.py` resolves the training adapter early via `DiffusionModelBase.get_class_by_name(architecture, algorithm, external_lib)` (also `prepare_processor_files`).
3. `DiffusionModelConfig` lookup: `workers/config/diffusion/model.py` calls `peek_class` to validate `(architecture, algorithm)`; `workers/engine/fsdp/diffusers_impl.py` uses `get_class` to build the module/scheduler and configure trainable params.
4. Rollout: `vllm_omni_async_server.py` resolves the custom pipeline class by dotted path from `VllmOmniPipelineBase.get_pipeline_path()`, which samples images/latents and emits log-prob trajectories into `diffusion_rollout_output.py` structures.
5. Training: engine forwards per denoising step through `utils.prepare_model_inputs` → `forward` → (flow/dance-GRPO) `scheduler.sample_previous_step` for `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)`; policy-loss computation lives in `trainer/diffusion/diffusion_algos.py`. Rewards are wired via the reward loop (`verl_omni/reward_loop/`) consuming generated images/prompts from the dataset class selected in the recipe config.

## Integration
- Consumed by: `verl_omni/trainer/` (diffusion + omni), `verl_omni/workers/` (engine, rollout, config), `verl_omni/agent_loop/`.
- Depends on: `diffusers` (ModelMixin/SchedulerMixin), `verl` utils, `vllm-omni` rollout engine.
- Key entry points: `model_base.py` registries; `utils.py` dispatchers; `trainer/main_diffusion.py`, `trainer/main_omni.py`.

### Pipeline directory table

| Directory | Architecture (registry key) | Algorithm | Responsibility |
|---|---|---|---|
| [schedulers/](schedulers/codemap.md) | — | — | Custom diffusion schedulers, incl. `FlowMatchSDEDiscreteScheduler` for SDE sampling with log-probs. |
| [bagel_flow_grpo/](bagel_flow_grpo/codemap.md) | `OmniBagelForConditionalGeneration` | `flow_grpo` | Bagel unified multimodal model with flow-GRPO; ships its own model defs (`bagel_model.py`, `bagel_sft_model.py`). |
| [boogu_image_flow_grpo/](boogu_image_flow_grpo/codemap.md) | `BooguImagePipeline` | `flow_grpo` | Boogu image model with flow-GRPO (internal/custom architecture). |
| [ltx2_flow_grpo/](ltx2_flow_grpo/codemap.md) | `LTX2Pipeline` | `flow_grpo` | LTX-2 audio-video (I2AV-conditioned) generation with flow-GRPO. |
| [minimax_h3_diffusion_nft/](../minimax_h3_diffusion_nft/) | `MiniMaxH3Pipeline` | `diffusion_nft` | MiniMax-H3 image model trained with Diffusion-NFT. |
| [qwen3_omni/](../qwen3_omni/) | `Qwen3OmniMoeForConditionalGeneration` | (verl loss_mode) | Qwen3-Omni MoE thinker training adapter + vLLM-Omni rollout topology (AR, drives `trainer/omni`). |
| [qwen_image_diffusion_nft/](../qwen_image_diffusion_nft/) | `QwenImagePipeline` | `diffusion_nft` | Qwen-Image with Diffusion-NFT (no vLLM-side sampling log-prob; teacher-free forward-process objective). |
| [qwen_image_dpo/](../qwen_image_dpo/) | `QwenImagePipeline` | `dpo` | Qwen-Image with pairwise DPO (adjacent chosen/rejected pairs sharing noise/timesteps). |
| [qwen_image_dual_grpo/](qwen_image_dual_grpo/codemap.md) | `QwenImagePipeline` | `dual_grpo` | Qwen-Image dual-policy GRPO: rollout adapter only, reusing the base flow-GRPO training adapter. |
| [qwen_image_edit_flow_grpo/](qwen_image_edit_flow_grpo/codemap.md) | `QwenImageEditPlusPipeline` | `flow_grpo` | Qwen-Image-Edit (image-conditioned I2I editing) with flow-GRPO. |
| [qwen_image_flow_grpo/](qwen_image_flow_grpo/codemap.md) | `QwenImagePipeline` | `flow_grpo` | Reference flow-GRPO pipeline: Qwen-Image t2i with SDE scheduler and True-CFG. |
| [qwen_image_mix_grpo/](qwen_image_mix_grpo/codemap.md) | `QwenImagePipeline` | `mix_grpo` | Qwen-Image with MixGRPO (mixed SDE/ODE sampling); thin reuse of flow-GRPO components. |
| [sd3_dpo/](../sd3_dpo/) | `StableDiffusion3Pipeline` | `dpo` | SD3 with pairwise DPO; training adapter only (no custom rollout pipeline). |
| [sd3_flow_grpo/](sd3_flow_grpo/codemap.md) | `StableDiffusion3Pipeline` | `flow_grpo` | SD3/SD3.5 t2i with flow-GRPO. |
| [wan22_dance_grpo/](wan22_dance_grpo/codemap.md) | `WanPipeline` | `dance_grpo` | Wan2.2 video (T2V/I2V) with DanceGRPO (ODE-style reversed sampling). |

(Directories without links are not covered by a dedicated codemap; they follow the same two-adapter pattern described above.)
