# verl_omni/models/diffusers/

## Responsibility
Adapters for diffusers-based diffusion models (Qwen-Image) used in diffusion RL post-training. Provides two idempotent monkey-patches against diffusers==0.38: one fixing the FlashAttention-3 varlen backend's non-contiguous attention-mask handling, and one fixing `QwenImageTransformer2DModel.forward`'s joint text/image attention mask layout under Ulysses sequence parallelism. Both are temporary workarounds pending upstream diffusers fixes (TODO comments).

## Design
- Patch-function pattern with idempotence markers: `apply_*()` functions attach a `_verl_omni_*_patched` attribute to the patched callable and set a lazy `_warned` flag so the one-time warning fires on first use.
- Version gating: both patches return early if `diffusers.__version__ < 0.38.0`.
- `flash_attention_3.py` reaches into diffusers internals (`_ad._AttentionBackendRegistry`, `_ad.AttentionBackendName._FLASH_3_VARLEN_HUB`) and replaces both the registry entry and `_ad._flash_attention_3_varlen_hub` module global.
- `qwen_image.py` wraps `QwenImageTransformer2DModel.forward`, reading `ulysses_degree` from the instance's `_parallel_config.context_parallel_config` (set by the rollout worker) at call time.

## Flow
1. `import verl_omni.models` → root `__init__.py` calls `apply_flash_attention_3_varlen_hub_fix()` unconditionally (universal fix).
2. When a Qwen-Image pipeline is built, `verl_omni/pipelines/model_base.py` (line ~72) imports and calls `apply_qwen_image_ulysses_mask_fix()`.
3. During rollout with `ulysses_degree > 1`: the patched `_patched_forward` rebuilds `encoder_hidden_states_mask` into interleaved `[txt_0, img_0, txt_1, img_1, ...]` order matching the post-all-to-all token layout, injects it into `attention_kwargs["attention_mask"]`, and delegates to the original forward.
4. During attention dispatch: the patched `_patched_flash_attention_3_varlen_hub` gathers valid K/V tokens via `attn_mask.flatten().nonzero()` (instead of the contiguous-prefix `key[b, :valid_len]`), packs Q/K/V with `cu_seqlens_*` from `_ad._prepare_for_flash_attn_or_sage_varlen`, calls the hub kernel, and unflattens the output.

## Integration
- Consumed by: `verl_omni.pipelines.model_base` (Qwen-Image patch), `tests/workers/test_diffusers_ulysses.py`, `tests/pipelines/test_qwen_image_edit_on_cpu.py`; FA3 fix applied by `verl_omni.models.__init__`.
- Depends on: `diffusers` (>=0.38 internals: `attention_dispatch`, `transformer_qwenimage`), `torch`, `packaging.version`.
- Key entry points: `apply_flash_attention_3_varlen_hub_fix` (flash_attention_3.py), `apply_qwen_image_ulysses_mask_fix` (qwen_image.py); both re-exported from `diffusers/__init__.py`.
