# verl_omni/pipelines/bagel_flow_grpo/

## Responsibility
Flow-GRPO pipeline for **BAGEL** (`OmniBagelForConditionalGeneration` architecture), a Mixture-of-Transformers (MoT) unified multimodal model. Because BAGEL is not a diffusers-native architecture, this directory is the largest non-diffusers pipeline: it ships its own FSDP-compatible training module (`BagelForTraining`), takes raw token IDs instead of prompt embeds, and applies 3-branch CFG with global renormalization matching the rollout pipeline. Drives `trainer/main_diffusion.py` (typically via `recipe/flowgrpo_trainer/`).

## Design
- `bagel_model.py` (~660 lines): `BagelTrainingConfig`, `BagelForTraining` MoT module with dual text/generation pathways, `BagelMoTAttention` (fused separate q/k/v projections), `TimestepEmbedder`, `PositionEmbedding`, `get_flattened_position_ids`, boundary tokens, fp32 QK-norm/RoPE.
- `bagel_sft_model.py` (~1400 lines): SFT-oriented variant — `BagelSFTConfig`, `BagelSFTOutput`, `BagelMLPConnector`, `BagelFrozenVAEEncoder`, checkpoint loading with `BagelCheckpointLoadReport`.
- `diffusers_training_adapter.py`: `@DiffusionModelBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")` class `BagelDiffusion`. Overrides `build_module` (loads `BagelForTraining.from_pretrained` instead of diffusers `AutoModel`), `configure_train_mode` (MoT train-flag workaround), `configure_trainable_params` (freeze everything except the `moe_gen` pathway, cast trainable params to fp32). Scheduler sigma setup via `common.setup_bagel_sigmas` (SD3-style time shift `BAGEL_TIMESTEP_SHIFT = 3.0`, identity scheduler shift).
- `common.py`: `BAGEL_FLOWGRPO_CFG_DEFAULTS` (`cfg_text_scale=4.0`, `cfg_renorm_type="global"` — pinned to flow_grpo's `train_bagel.py` to keep rollout↔training aligned), `bagel_time_shift`, `maybe_to_cpu`.
- `vllm_omni_rollout_adapter.py`: `@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")` class `BagelPipelineWithLogProb(BagelPipeline)` (from `vllm_omni.diffusion.models.bagel.pipeline_bagel`). Wraps the SDE scheduler in `_BagelSchedulerAdapter`, applies per-request **SDE windowing** (noise only on a contiguous random subset of denoising steps, seeded RNG), and overrides `load_weights` to route `transformer.`-prefixed actor weights through `language_model.load_weights` for separate q/k/v → fused `qkv_proj` remapping.

## Flow
1. Rollout: `BagelPipelineWithLogProb` denoises with the adapted `FlowMatchSDEDiscreteScheduler` over the SDE window, recording latents/timesteps/log-probs (extracted via `_extract_bagel_trajectory` from `DiffusionOutput.trajectory_*` fields) into `pipelines.diffusion_rollout_output.rollout_output`.
2. Training: `BagelDiffusion.build_module` loads the MoT module; micro-batches carry BAGEL-native `prompt_token_ids` (padded by `_prompt_token_ids_to_batch`); latent position IDs come from `_get_latent_pos_ids` using `latent_patch_size * vae_downsample`.
3. Forward + reverse sampling through `FlowMatchSDEDiscreteScheduler.sample_previous_step` yields `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)` for the flow-GRPO loss in `trainer/diffusion/diffusion_algos.py`; rewards wired via the standard reward loop on generated images.

## Integration
- Consumed by: diffusion trainer/engine via the `(OmniBagelForConditionalGeneration, flow_grpo)` registry keys; `NonDiffusersModelBase` pattern for non-diffusers module loading.
- Depends on: `pipelines/schedulers.FlowMatchSDEDiscreteScheduler`, `pipelines.model_base`, `pipelines.non_diffusers_model_base`, vllm-omni `BagelPipeline`.
- Key entry points: `BagelDiffusion`, `BagelPipelineWithLogProb`, `BagelForTraining`, `setup_bagel_sigmas`.
- Sibling differences: only GRPO pipeline that (a) registers a fully custom model implementation rather than a diffusers model, (b) uses token IDs instead of text embeddings, (c) restricts training to the `moe_gen` pathway.
