# verl_omni/models/

## Responsibility
Model-integration adapters that bridge upstream Hugging Face model libraries (`diffusers`, `transformers`) to the verl-omni training/rollout engines via targeted monkey-patches. Each patch fixes an upstream bug or adapts a model class to verl's training contract (FSDP init, vLLM weight sync, PEFT LoRA, Ulysses sequence parallelism). The package root eagerly applies the universal diffusers fix on import.

## Design
- Patch-function pattern: every adapter is an idempotent `apply_*()` function guarded by a marker attribute (e.g. `_verl_omni_ulysses_mask_patched`, `_verl_omni_fa3_varlen_patched`, `_EXPERTS_UNFUSE_APPLIED`) so repeat imports never double-patch.
- Version gating with `packaging.version.parse(diffusers.__version__)` so patches no-op on older diffusers.
- Import-time side effects allowed so modules double as verl `external_lib` targets.

## Flow
1. `import verl_omni.models` → `__init__.py` imports from `.diffusers` and immediately calls `apply_flash_attention_3_varlen_hub_fix()` (universal FA3 varlen fix).
2. Model-specific patches (`apply_qwen_image_ulysses_mask_fix`, Qwen3-Omni thinker patches) are invoked by the consuming worker/pipeline when that model is loaded.
3. Patched classes are then used transparently by the training engine.

## Integration
- Consumed by: `verl_omni.pipelines.model_base` (calls `apply_qwen_image_ulysses_mask_fix`), tests under `tests/workers/`, `tests/pipelines/`; the transformers subpackage is applied on import as a verl `external_lib` target.
- Depends on: `diffusers`, `transformers`, `peft`, `verl.utils.model`, `verl.utils.tokenizer`, `verl.utils.vllm.patch`.
- Key entry points: `verl_omni.models.__init__`, `apply_flash_attention_3_varlen_hub_fix`, `apply_qwen_image_ulysses_mask_fix`, `apply_qwen3_omni_thinker_patches`.

| Subpackage | Purpose | Key module | Codemap |
|---|---|---|---|
| `diffusers/` | Monkey-patches for diffusers==0.38 diffusion models (Qwen-Image transformer, FA3 varlen attention) | `qwen_image.py`, `flash_attention_3.py` | [diffusers/codemap.md](diffusers/codemap.md) |
| `transformers/` | (DEPRECATED, removal in v0.3.0) Qwen3-Omni Thinker adapter patches for AR/omni training under the legacy trainer | `qwen3_omni_thinker.py` | [transformers/codemap.md](transformers/codemap.md) |
