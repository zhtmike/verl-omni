# verl_omni/utils/

## Responsibility
Shared utility subpackage for verl-omni: dataset loading/preprocessing, visual reward scorers, diffusion MFU accounting, the vLLM-Omni rollout bridge, and misc training utilities (config validation, FSDP/LoRA export, metrics aggregation, W&B media logging, attention-backend checks). The package `__init__.py` is empty (license header only); each child module is imported directly.

## Design
Flat utility modules plus three focused subpackages (`dataset/`, `mfu/`, `reward_score/`, `vllm_omni/`). Helper style: small pure functions and thin adapter classes that extend upstream veRL classes rather than wrapping them.

## Flow
1. Trainer entrypoints call `config.validate_config` before launching.
2. Data pipeline uses `dataset.create_rl_dataset` / offline DPO datasets; reward managers use `reward_score` scorers.
3. Rollout workers use `vllm_omni.VLLMOmniHijack` to sync LoRA tensors; MFU is computed by `mfu.DiffusionFlopsCounter`.

## Integration
- `config.py` — `validate_config`: fail-fast checks for `trainer.resume_mode` / `total_training_steps`, used by trainer entrypoints.
- `dataset/` — parquet/jsonl RLHF + offline DPO dataset classes, multimodal media loading; see [dataset/codemap.md](dataset/codemap.md).
- `diffusion_attention.py` — `fa3_available`, `rollout_fa3_available`, `fallback_fa3_if_unavailable`, `validate_attention_consistency`: FlashAttention-3 capability probing and actor/rollout attention-backend consistency checks.
- `fs.py` — `resolve_model_local_dir`: resolves model paths to a local directory (optional /dev/shm).
- `fsdp_utils.py` — FSDP2 helpers: `fsdp_summon_full_params`, `collect_lora_params`, `export_fsdp_lora_adapter`, `split_fused_moe_lora_targets` for exporting FSDP-sharded LoRA to PEFT adapters.
- `mfu/` — diffusion FLOPs/MFU computation; see [mfu/codemap.md](mfu/codemap.md).
- `metrics_utils.py` — `_MetricMeanStats`, `GroupedMetricMean`: weighted mean aggregation of rollout metrics, optionally grouped by an attribute (e.g. task type).
- `reward_score/` — visual/audio reward scorers; see [reward_score/codemap.md](reward_score/codemap.md).
- `tracking.py` — `wrap_val_samples_for_wandb`, `log_wandb_media`, `batch_items`: converts validation image/audio/video tensors to W&B media payloads.
- `vllm_omni/` — vLLM-Omni bridge; see [vllm_omni/codemap.md](vllm_omni/codemap.md).
- `profiler/` — exists as an empty directory (no files, no codemap.md); reserved placeholder for profiling utilities.
