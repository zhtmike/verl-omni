# verl_omni/workers/engine/veomni/

## Responsibility
veOmni-backed training engine for diffusion models: `VeOmniDiffusionEngine`, registered as `@EngineRegistry.register(model_type="diffusion_model", backend=["veomni"], device=["cuda"])`. Delegates model build/parallelization/optimizer construction to veOmni's `DiTTrainer`/`BaseTrainer` while exposing verl's `BaseEngine` protocol; FSDP2-only, no LoRA support.

## Design
- Single class `VeOmniDiffusionEngine(BaseEngine)` in `diffusion_impl.py`; `VeOmniDiffusionEngineConfig`/`VeOmniDiffusionOptimizerConfig` (defined in `config/diffusion/actor.py`) pin `strategy="veomni"`, `fsdp_mode="fsdp2"`, and reject Ulysses SP.
- **Constructor-time guard**: raises `NotImplementedError` if `lora_rank > 0` or `lora_adapter_path` set — use the fsdp/fsdp2 diffusion backend for LoRA runs.
- **veOmni argument bridge**: `_build_veomni_dit_args` assembles `VeOmniDiTArguments` from verl-omni configs — `FSDPConfig(fsdp_mode="fsdp2", reshard_after_forward, forward_prefetch, mixed_precision)`, `AcceleratorConfig(dp_replicate_size, dp_shard_size, ep_size, ulysses_size, offload_config)`, `OptimizerConfig` (mapped from `VeOmniDiffusionOptimizerConfig` incl. `lr_min`/`lr_start`/`lr_decay_ratio`), `GradientCheckpointingConfig`, `OpsImplementationConfig` (via `_build_ops_config` reflecting dataclass fields), `training_task="offline_training"`.
- **Trainer reuse without full init**: `_build_veomni_dit_trainer` uses `DiTTrainer.__new__` + manual `BaseTrainer` attribute wiring to skip dataset/CLI setup; `_build_model_optimizer` calls `BaseTrainer._build_model` → `_build_parallelized_model` → `_build_optimizer` → `_build_lr_scheduler` → `_build_training_context`, exposing `module`, `optimizer`, `lr_scheduler`, `model_fwd_context`/`model_bwd_context`.
- Parallel layout via `veomni.distributed.parallel_state.init_parallel_state(dp_size, dp_replicate_size, dp_shard_size, expert_parallel_size, ulysses_size=1, dp_mode="fsdp2")`; DP accessors read `parallel_state.get_parallel_state()` (`dp_rank`, `dp_size`, `dp_group`, `sp_rank`).
- Mode contexts: local `EngineTrainModeCtx` (train + `optimizer_zero_grad` on exit) and `EngineEvalModeCtx` (eval + `module.reshard()` on exit when fsdp_size > 1), both extending verl's `BaseEngineCtx`.

## Flow
1. `initialize()` → `_build_model_optimizer` (above) → build diffusion `scheduler` via `build_scheduler(model_config)` → wrap in verl `FSDPCheckpointManager` → `to("cpu")` if offload enabled.
2. Training mirrors the FSDP diffusion path: `forward_backward_batch(data, loss_function, forward_only)` → `prepare_micro_batches` → per micro-batch × per timestep `forward_step` (`prepare_model_inputs` with nested-embed unpad/SP pad → `forward_and_sample_previous_step` → loss function) → `loss.backward()` → `optimizer_step` (grad clip via `module.clip_grad_norm_`, skip non-finite) → `postprocess_batch_func` aggregates `model_output`/`loss`/`metrics`.
3. Offload: `to(device)` uses veOmni's `load_model_to_gpu`/`offload_model_to_cpu` and `load_optimizer`/`offload_optimizer`; activation offload flags feed `OffloadConfig`.
4. Weight sync: `get_per_tensor_param(**kwargs)` — raises for LoRA; otherwise loads model to GPU, takes `module.state_dict()`, `convert_weight_keys`, `full_tensor()` per DTensor, casts floating tensors to `model_dtype`, yields `("transformer." + name, tensor)` with `peft_config=None`.
5. Checkpoints: `save_checkpoint`/`load_checkpoint` via `FSDPCheckpointManager` with GPU restore / re-offload and barriers.

## Integration
- Consumed by: `EngineRegistry` → `TrainingWorker` in `engine_workers.py` when `actor.strategy == "veomni"`.
- Depends on: `veomni.distributed` (parallel_state, offloading), `veomni.trainer.dit_trainer` (`DiTTrainer`, `VeOmniDiTArguments`), `veomni.arguments` (config dataclasses), `verl.workers.engine.base`, `verl.workers.engine.utils`, `verl_omni.pipelines.utils`, `verl_omni.workers.config`.
- Key entry points: `VeOmniDiffusionEngine`, `EngineTrainModeCtx`, `EngineEvalModeCtx`.
