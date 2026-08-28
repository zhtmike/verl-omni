# verl_omni/utils/vllm_omni/

## Responsibility
Bridge between verl-omni's FSDP actor and the vLLM-Omni diffusion rollout engine: enables in-memory LoRA tensor synchronization so rollout workers can load freshly trained actor weights without writing adapter files to disk.

## Design
Monkey-patching adapter layer. `OmniTensorLoRARequest` extends vllm-omni's `OmniLoRARequest` with `peft_config` and `lora_tensors` fields. `VLLMOmniHijack` is an idempotent static patcher (`_patched` guard) that first applies verl's `VLLMHijack.hijack()`, then replaces `vllm_omni.diffusion.lora.manager.DiffusionLoRAManager._load_adapter` with a tensor-aware reimplementation.

## Flow
1. Worker init calls `VLLMOmniHijack.hijack()`; subsequent calls are no-ops.
2. Trainer pushes `OmniTensorLoRARequest(lora_tensors=..., peft_config=...)` to the rollout engine.
3. Patched `_load_adapter` optionally translates tensors/engine layout via the pipeline's `map_lora_update_to_engine` mapper (needed for engines whose module names differ from diffusers names, e.g. MiniMax H3 fused DiT), builds a `PEFTHelper` from the dict config, and constructs `LoRAModel.from_lora_tensors` (CPU, matching vLLM behavior) instead of `from_local_checkpoint`.
4. Each LoRA module is `optimize()`d (internal scaling) and returned to the manager for binding to the diffusion pipeline; file-path-based `OmniLoRARequest`s keep the original vLLM path resolution (`get_adapter_absolute_path` → `PEFTHelper.from_local_dir` → `LoRAModel.from_local_checkpoint`).

## Integration
- Consumed by: rollout workers in `verl_omni/workers/` (diffusion rollout) on every worker construction / LoRA sync.
- Depends on: `vllm` (`LoRAModel`, `PEFTHelper`, `get_adapter_absolute_path`, verl's `VLLMHijack`), `vllm_omni.diffusion.lora.manager.DiffusionLoRAManager`, `vllm_omni.lora.request.OmniLoRARequest`.
- Key entry points: `OmniTensorLoRARequest`, `VLLMOmniHijack.hijack()` (both re-exported from `verl_omni/utils/vllm_omni/__init__.py`).
- Files: `utils.py` (all logic), `__init__.py` (re-exports).
