# verl_omni/utils/mfu/

## Responsibility
Model FLOPs Utilization (MFU) accounting for diffusion training: estimates per-step FLOPs from transformer configs and batch metadata, and aggregates them across data-parallel ranks.

## Design
Registry pattern: `DiffusionModelFlops` is the abstract base (dense + attention FLOPs estimation, `get_latent_seqlens`/`get_prompt_seqlens` from batch data); architectures are registered via the `@register_diffusion_architecture(name)` decorator into a class map resolved by `DiffusionFlopsCounter.architecture_cls`. `__init__.py` imports `qwen_image` for its decorator side effect so `QwenImageFlops` is registered on import, and re-exports the public API.

## Flow
1. Construct `DiffusionFlopsCounter(architecture, transformer_config)`.
2. `collect_diffusion_flops_meta(data, ...)` extracts `latent_seqlens` / `prompt_seqlens` per sample from the batch (via the architecture subclass).
3. `allgather_diffusion_flops_meta(meta, dp_group)` syncs metadata across the DP group.
4. `counter.estimate_flops(meta)` dispatches to the registered subclass (e.g. `QwenImageFlops.estimate_flops`), which combines `compute_dense_flops(params_per_token, total_tokens)` and `compute_attention_flops` (via `sum_seqlen_squared`) over `get_forward_passes_per_step` diffusion steps.
5. Divide by `get_device_peak_tflops()` (hardware table lookup by GPU/NPU name) and step time to get MFU.

## Integration
- Consumed by: diffusion trainer metric reporting (MFU logging).
- Depends on: torch distributed (DP group all-gather), device-name introspection for peak-TFLOPS tables.
- Key entry points: `DiffusionFlopsCounter`, `register_diffusion_architecture`, `QwenImageFlops`, `get_device_peak_tflops`, `get_forward_passes_per_step`, `collect_diffusion_flops_meta`, `allgather_diffusion_flops_meta`.
- Files: `diffusion_flops_counter.py` (base class, registry, counter, meta collection), `qwen_image.py` (`QwenImageFlops` — latent seqlens from packed image latents, Qwen-Image-specific FLOPs formula). Add new architectures by subclassing `DiffusionModelFlops` and applying `register_diffusion_architecture`.
