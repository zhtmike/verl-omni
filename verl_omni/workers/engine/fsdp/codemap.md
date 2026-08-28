# verl_omni/workers/engine/fsdp/

## Responsibility
FSDP/FSDP2 training engines for diffusion models (`diffusers_impl.py`) and omni models (`omni_impl.py`). Handles model construction (diffusers AutoModel or pipeline-registry loaders), FSDP1/FSDP2 wrapping with mixed precision and offload, per-timestep forward/backward, LoRA adapter training, checkpointing, and per-tensor weight export for rollout sync.

## Design
- `DiffusersFSDPEngine(LoRAAdapterMixin, BaseEngine, ABC)` — abstract base; concrete hooks: `forward_backward_batch`, `prepare_model_inputs`, `prepare_model_outputs`, `forward_step`.
  - Registered subclasses: `PPODiffusersFSDPEngine` (`diffusion_model`; iterates `all_timesteps` steps, `forward_and_sample_previous_step` → `{log_probs, prev_sample_mean, std_dev_t, sqrt_dt}`), `DPODiffusersFSDPEngine` (`diffusion_dpo_model`; one-shot flow-matching step over `latents_clean` via `prepare_noisy_latents`, returns `noise_pred` + context), `NFTDiffusersFSDPEngine` (`diffusion_nft_model`; forward-process noising `xt=(1-t)x0+t·noise` over `train_timesteps`).
- **Sharding/offload**: `_init_device_mesh` builds the FSDP mesh (`create_device_mesh`) and an optional Ulysses SP mesh `(dp, ring, ulysses)` requiring diffusers ≥ 0.38; `_build_fsdp_module` applies FSDP1 (`FSDP`, `MixedPrecision`, forced `CPUOffload` only for `forward_only`) or FSDP2 (`apply_fsdp2`, `MixedPrecisionPolicy`, optional `CPUOffloadPolicy(pin_memory=True)` tracked via `_uses_fsdp2_cpu_offload_policy`); state-dict type FULL vs SHARDED per world size. fp32 "islands" preserved via `_keep_in_fp32_modules`/`preserve_fp32_modules`.
- `LoRAAdapterMixin` (`../lora_adapter_mixin.py`): builds/loads named PEFT adapters (`policy_state_adapters`), `_adapter_state_context` (FSDP summon full params + offload round-trip), `copy_adapter`/`ema_update_adapter` for `default→old` policy states.
- `OmniFSDPEngine(FSDPEngineWithLMHead)` — omni variant: loads via `AutoModelForMultimodalLM` + `OmniModelBase.get_class_by_name(...).configure_model`, rejects `use_liger`/`use_fused_kernels`, supports LoRA merge via `merged_lora_context` and optional QAT quantization on weight export.

## Flow
1. `initialize()` → `_build_model_optimizer`: `_build_module` (registry loader `_build_module_from_registry` else diffusers `AutoModel.from_pretrained` with meta-tensor init context) → LoRA or `configure_trainable_params` → Ulysses `enable_parallelism(ContextParallelConfig)` → `build_scheduler` → `_build_fsdp_module` → `_build_optimizer`/`_build_lr_scheduler` (skipped when `forward_only`).
2. Train: `train_mode()` context (`EngineTrainModeCtx`) → `train_batch` (from `TrainingWorker`) → `forward_backward_batch` → `prepare_micro_batches` per DP group → per micro-batch × per timestep `forward_step` (loss via injected `loss_function`, e.g. `diffusion_loss`) → `loss.backward()` → `optimizer_step` (clip via `module.clip_grad_norm_` / `fsdp2_clip_grad_norm_`, skip on non-finite) → `lr_scheduler_step`; results aggregated by `postprocess_batch_func`.
3. Weight sync export: `get_per_tensor_param(layered_summon, base_sync_done, adapter_name)` — load model to GPU (skipped under FSDP2 CPUOffloadPolicy), gather LoRA via `collect_lora_params` (verl-omni fixed version) or `module.state_dict()`, `convert_weight_keys`, `full_tensor().to(bf16)` per DTensor, prefix `transformer.` for the rollout engine; returns `(per_tensor_param, peft_config_dict)`.
4. Checkpoints: `save_checkpoint`/`load_checkpoint` delegate to verl's `FSDPCheckpointManager` with GPU/offload round-trips.

## Integration
- Consumed by: `EngineRegistry` → `TrainingWorker` (`engine_workers.py`); `DiffusionDetachActorWorker` snapshots `engine.module`.
- Depends on: `verl.utils.fsdp_utils` (`apply_fsdp2`, `fsdp2_load_full_state_dict`, offload/load helpers), `verl.workers.engine.fsdp.utils` (`create_device_mesh`, `get_sharding_strategy`), `verl.workers.engine.utils` (`prepare_micro_batches`, `enable_full_determinism`), `verl_omni.pipelines.utils` (`build_scheduler`, `forward`, `forward_and_sample_previous_step`, `prepare_model_inputs`, `prepare_noisy_latents`), `verl_omni.utils.fsdp_utils.collect_lora_params`, `verl_omni.workers.config.DiffusionModelConfig`.
- Key entry points: `PPODiffusersFSDPEngine`, `DPODiffusersFSDPEngine`, `NFTDiffusersFSDPEngine`, `OmniFSDPEngine`.
