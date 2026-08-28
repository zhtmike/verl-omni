# verl_omni/workers/engine/

## Responsibility
Package root for verl-omni's training engines — the `BaseEngine` implementations registered in verl's `EngineRegistry` for verl-omni model types. `__init__.py` eagerly imports every engine implementation so the `@EngineRegistry.register` decorators run at import time, making engines resolvable by `(model_type, backend, device)`.

## Design
- **Registry-driven plugins**: engines self-register via `@EngineRegistry.register(model_type=..., backend=[...], device=[...])` on verl's `verl.workers.engine.base.EngineRegistry`; `TrainingWorker` resolves them with `EngineRegistry.new(...)`. No factory code lives here — just imports.
- **Shared mixin**: `lora_adapter_mixin.py` provides `LoRAAdapterMixin`, backend-agnostic named-PEFT-adapter lifecycle used by the diffusion FSDP engines (`_build_lora_module`, `use_adapter`, `disable_adapter`, `copy_adapter`, `ema_update_adapter`).
- **Engine families**:
  - `fsdp/`: `DiffusersFSDPEngine` (ABC base) + `PPODiffusersFSDPEngine`, `DPODiffusersFSDPEngine`, `NFTDiffusersFSDPEngine` (registered for `diffusion_model` / `diffusion_dpo_model` / `diffusion_nft_model`, backends `fsdp`/`fsdp2`, devices cuda/npu) and `OmniFSDPEngine` (`omni_model`, extending verl's `FSDPEngineWithLMHead`).
  - `veomni/`: `VeOmniDiffusionEngine` (`diffusion_model`, backend `veomni`, cuda), imported in a `try/except ImportError` since veOmni is optional.

## Flow
1. `TrainingWorker.__init__` (in `engine_workers.py`) calls `EngineRegistry.new(model_type, backend=engine_config.strategy, ...)` — this package's import side effects are what populated the registry.
2. `reset()` → `engine.initialize()` → the concrete engine builds model/optimizer/scheduler and (for FSDP) wraps with FSDP/FSDP2; see sub-maps for details.
3. Trainer calls the `BaseEngine` protocol: `train_mode`/`eval_mode` context managers, `train_batch`/`infer_batch`, `optimizer_step`, `lr_scheduler_step`, `to(device)`, `save_checkpoint`/`load_checkpoint`, `get_per_tensor_param` (weight sync export), `get_data_parallel_*`, `is_mp_src_rank_with_outputs`.

## Integration
- Consumed by: `verl_omni/workers/engine_workers.py` (`TrainingWorker`, `ActorRolloutRefWorker`); `DiffusionDetachActorWorker` snapshots via `self.actor.engine`.
- Depends on: `verl.workers.engine.base` (`BaseEngine`, `BaseEngineCtx`, `EngineRegistry`), `verl_omni.pipelines` (model adapters, `build_scheduler`, `forward*` helpers), `verl_omni.workers.config`, `verl_omni.utils.fsdp_utils.collect_lora_params`.
- Sub-maps: `fsdp/codemap.md`, `veomni/codemap.md`.
- Key entry points: `PPODiffusersFSDPEngine`, `OmniFSDPEngine`, `VeOmniDiffusionEngine`, `LoRAAdapterMixin`.
