# verl_omni/workers/config/diffusion/

## Responsibility
Config dataclasses for diffusion-model RL post-training: actor/loss/engine/optimizer configs, rollout + pipeline sampling configs, and on-policy distillation teacher configs. Validated at construction and consumed by the diffusion FSDP/veOmni engines and the vLLM-Omni diffusion rollout server.

## Design
- `actor.py`:
  - `DiffusionLossConfig(BaseConfig)` — loss hyperparams (`loss_mode` in {`flow_grpo`, `flow_dppo`, `grpo_guard`, `diffusion_nft`, `dpo`, `dance_grpo`, `distill_kl`, `distill_fm_mse`}, `clip_ratio`, `adv_clip_max`, `mix_beta`, `ref_kl_coef`, `dpo_beta`), validated in `__post_init__`.
  - `VeOmniDiffusionEngineConfig(EngineConfig)` — strategy fixed to `"veomni"`, FSDP2-only, mixed-precision dtype fields, per-op eager implementations, rejects `ulysses_parallel_size != 1`.
  - `VeOmniDiffusionOptimizerConfig(OptimizerConfig)` — adamw defaults, `lr_scheduler_type` ∈ {constant, linear, cosine}.
  - `DiffusionActorConfig(BaseConfig)` — `ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu` (MISSING), `diffusion_loss`, KL/distill flags, `checkpoint`/`optim`/`engine` blocks, `global_batch_info`, `rollout_correction` (bypass-mode RC), `profiler`.
  - `FSDPDiffusionActorConfig` / `VeOmniDiffusionActorConfig` — wire `self.engine` to `fsdp_config` / `veomni_config` and sync `strategy` (FSDP variant uses `object.__setattr__` so fsdp2 reaches the engine config).
- `model.py`: `DiffusionModelConfig(BaseConfig)` — `path`/`local_path`, `architecture` (auto-read from `model_index.json` `_class_name`), `transformer_config` (for MFU), `attn_backend` ∈ {`native`, `_native_npu`, `_flash_3_varlen_hub`}, LoRA fields (`lora_rank`, `target_modules`, `lora_adapter_path`, `policy_state_adapters`, `lora_dtype`), `pipeline: DiffusionPipelineConfig`, `algo: DiffusionRolloutAlgoConfig`, `fsdp_layer_prefixes`, `transformer_subfolder`; loads tokenizer/processor/extra tokenizers in `__post_init__`.
- `rollout.py`: `DiffusionRolloutAlgoConfig` (SDE rollout: `sde_type`, `sde_window_size/range`, MixGRPO `sample_strategy`), `DiffusionPipelineConfig` (height/width/`num_inference_steps`/`true_cfg_scale`/`num_frames`/`frame_rate`/`task`/`aspect_ratio`/`video_flow_shift`), `DiffusionSamplingConfig` (validation-only sampling), `DiffusionRolloutConfig` (TP/DP/text-encoder-TP sizes, `rollout_attn_backend` default `FLASH_ATTN_3_HUB`, `step_execution`, prompt-embed cache, `checkpoint_engine`, `rollout_adapter` ∈ {default, old}, `to_vllm_omni_attention_config()`).
- `distillation.py`: `DiffusionDistillationTeacherModelConfig` (`key`, `model_path`, `world_size`) and `DiffusionDistillationConfig` (`enabled`, `nnodes`==0 ⇒ teachers colocated with actor, `teacher_models` dict re-keyed by `teacher_key`, resource-pool size validation).

## Flow
1. OmegaConf `actor_rollout_ref.*` → `omega_conf_to_dataclass` produces `DiffusionActorConfig`/`DiffusionRolloutConfig` in `ActorRolloutRefWorker.init_model`.
2. `DiffusionActorConfig.engine` selects the backend factory; `model_config` reaches `DiffusersFSDPEngine`/`VeOmniDiffusionEngine` and `DiffusionFlopsCounter`.
3. `DiffusionRolloutConfig` configures `vLLMOmniHttpServer` (`_init_config`) and the `DiffusionOutput` generation params; `diffusion.py` distillation config drives teacher `TrainingWorker` construction in `engine_workers.build_teacher_training_config`.

## Integration
- Consumed by: `verl_omni/workers/engine_workers.py`, `engine/fsdp/diffusers_impl.py`, `engine/veomni/diffusion_impl.py`, `rollout/vllm_rollout/vllm_omni_async_server.py`, `utils/losses.py` (`diffusion_loss`).
- Depends on: `verl.base_config.BaseConfig`, `verl.workers.config` (`EngineConfig`, `FSDPEngineConfig`, `OptimizerConfig`, `MtpConfig`, `CheckpointEngineConfig`, `AgentLoopConfig`, `DisaggregationConfig`), `verl.trainer.config` (`CheckpointConfig`, `RolloutCorrectionConfig`).
- Key entry points: `DiffusionActorConfig`, `DiffusionModelConfig`, `DiffusionRolloutConfig`, `DiffusionDistillationConfig` (all re-exported from `verl_omni.workers.config`).
