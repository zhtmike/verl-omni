# verl_omni/pipelines/sd3_dpo/

## Responsibility
Wires Stable Diffusion 3 / SD3.5 into the diffusion DPO algorithm (`algorithm="dpo"`), supporting both online and offline preference training. This is the smallest DPO pipeline: only the training-side diffusers adapter exists; rollout is handled by offline preference data or external sampling, and the pairwise DPO objective lives in `verl_omni/workers/utils/losses` / `DPOLoss`.

## Design
- Registration: `StableDiffusion3DPO` via `@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="dpo")` in `diffusers_training_adapter.py`. No `VllmOmniPipelineBase` registration in this directory (unlike `qwen_image_dpo`).
- `build_scheduler` loads `FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")` via `_build_sd3_scheduler`; `set_timesteps` is intentionally a no-op — DPO flow-matching samples timesteps from the full `num_train_timesteps` schedule (logit-normal over ~1000 steps) rather than an inference schedule.
- `prepare_model_inputs` requires `prompt_embeds_mask` and SD3-specific `pooled_prompt_embeds` / `negative_pooled_prompt_embeds` from the micro-batch (validated with a `KeyError` if `pooled_projections` is missing); `build_transformer_inputs` assembles `SD3Transformer2DModel` kwargs including `joint_attention_kwargs={"attention_mask": ...}`.
- `forward` runs one transformer call and applies plain (non-normalizing) CFG when `guidance_scale > 1`: `negative + s * (cond - negative)`.

## Flow
1. Offline entry: `python3 -m verl_omni.trainer.main_diffusion --config-name=offline_dpo_trainer` with `data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_dpo_dataset`, `name=OfflineDPODataset`, `collate_fn=offline_dpo_collate_fn` (e.g. `examples/dpo_trainer/sd35/run_sd35_medium_offline_dpo_lora.sh`); preference pairs come from pre-generated parquet data prepared by `examples/dpo_trainer/data_process/prepare_offline_dpo.py`.
2. Online entry: same `main_diffusion` trainer with `algorithm.sample_source=online` (rollout pairs grouped into chosen/rejected by prompt).
3. Training: caller passes already-noised latents and sampled training timesteps → `prepare_model_inputs` builds positive and (optionally CFG) negative transformer inputs → `forward` returns predicted noise for the pairwise DPO loss (reference model + `-logsigmoid(beta * Δ)` in `DPOLoss`).

## Integration
- Consumed by: `verl_omni/pipelines/__init__.py`; `main_diffusion.py` → `DirectPreferenceRayTrainer`.
- Depends on: `verl_omni.pipelines.model_base.DiffusionModelBase`, `verl_omni.workers.config.DiffusionModelConfig`, diffusers (`FlowMatchEulerDiscreteScheduler`, `ModelMixin`, `SchedulerMixin`), the DPO loss registered as `@register_diffusion_loss("dpo")` in `verl_omni/trainer/diffusion/diffusion_algos.py`, and `OfflineDPODataset` for offline data.
- Key entry points: `StableDiffusion3DPO`, `build_transformer_inputs`, `_build_sd3_scheduler`.
- Differences from `qwen_image_dpo`: SD3 needs pooled prompt projections (`pooled_projections`); no rollout-side vLLM-Omni pipeline adapter here; CFG in `forward` is the plain linear form (no norm normalization); `set_timesteps` is a no-op because training timesteps are sampled from the full schedule.
