# verl_omni/workers/config/

## Responsibility
Config dataclasses shared across the verl-omni worker stack. Re-exports the diffusion-specific and omni-specific config families into a single namespace (`verl_omni.workers.config`), so `engine_workers.py`, engines, and rollout servers can do one import (`DiffusionActorConfig`, `DiffusionModelConfig`, `OmniModelConfig`, ...). All classes extend verl's `BaseConfig` (a pydantic-flavored dataclass base with `omega_conf_to_dataclass` support, `_mutable_fields`, and `__post_init__` validation).

## Design
- **Package as namespace**: `__init__.py` merges `diffusion.__all__` + `omni.__all__` via star-imports; no logic.
- **Layered inheritance** (diffusion side): `DiffusionActorConfig` (strategy-agnostic) ← `FSDPDiffusionActorConfig` / `VeOmniDiffusionActorConfig`, with `engine` field wired to the matching `FSDPEngineConfig` / `VeOmniDiffusionEngineConfig` in `__post_init__`.
- **Fat model configs**: `DiffusionModelConfig` and `OmniModelConfig` do I/O in `__post_init__` — resolve `local_path` (`resolve_model_local_dir`), load HF tokenizer/processor, auto-detect `architecture` from `model_index.json` / `config.json`, validate LoRA settings through `DiffusionModelBase.validate_lora_config`.
- **MISSING sentinels** (`omegaconf.MISSING`) for required fields (`path`, `strategy`, `rollout_n`).

## Flow
1. Trainer YAML (OmegaConf DictConfig) → `omega_conf_to_dataclass(config.actor, dataclass_type=DiffusionActorConfig)` etc. in `engine_workers.py::init_model`.
2. `__post_init__` runs validation (allowed `loss_mode`, `trainer_type`, `attn_backend`, tp-size divisibility) and derives derived fields (`engine`, `local_path`, `hf_config`, tokenizers).
3. Configs travel into engine constructors (`DiffusersFSDPEngine`, `OmniFSDPEngine`, `VeOmniDiffusionEngine`) and the vLLM-Omni server, which read model/lora/engine/optimizer sub-blocks.

## Integration
- Consumed by: `verl_omni/workers/engine_workers.py`, all engines in `engine/`, `rollout/vllm_rollout/vllm_omni_async_server.py`, trainer loss builders (`utils/losses.py`).
- Depends on: `verl.base_config.BaseConfig`, `verl.workers.config` (`FSDPEngineConfig`, `FSDPOptimizerConfig`, `FSDPActorConfig`, `RolloutConfig` sub-blocks, `MtpConfig`, `CheckpointEngineConfig`, `DisaggregationConfig`), `verl_omni.utils.fs.resolve_model_local_dir`.
- Sub-maps: `diffusion/codemap.md`, `omni/codemap.md`.
- Key entry points: `verl_omni.workers.config.DiffusionActorConfig`, `.DiffusionModelConfig`, `.OmniModelConfig` (the types `engine_workers` imports).
