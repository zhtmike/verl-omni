# verl_omni/models/transformers/

## Responsibility
(DEPRECATED — will be removed in v0.3.0; superseded by the V1 trainer `verl_omni.trainer.main_omni` which handles Qwen3-Omni natively.) Adapters for transformers-based AR/omni models, specifically Qwen3-Omni Thinker. Its single module `qwen3_omni_thinker.py` monkey-patches the HF model class, verl's tokenizer/processor factories, PEFT, and vLLM weight-loading so the legacy trainer can FSDP-train Qwen3OmniMoeThinker (including LoRA on MoE experts under transformers 5.x).

## Design
- Patch-function pattern, all idempotent; orchestrated by `apply_qwen3_omni_thinker_patches()` which runs at module import (so the module works as a verl `external_lib` target) and calls five sub-patches. ImportError guards make each patch a no-op if the relevant library/class is absent; a module-level `FutureWarning` flags deprecation.
- `_register_qwen3_omni_automodel`: maps `Qwen3OmniMoeForConditionalGeneration` → `AutoModelForMultimodalLM` in verl's `_architecture_to_auto_class`; redirects `forward`/`get_input_embeddings`/`set_input_embeddings` to `self.thinker`; fixes `_no_split_modules` to the real `Qwen3OmniMoeThinkerTextDecoderLayer`/`Qwen3OmniMoeVisionBlock`; adds `_verl_strip_modules = ["talker", "code2wav", "code_predictor"]` for Thinker-only training; forces `tie_word_embeddings=False` on `Qwen3OmniMoeConfig` via a `_FalseTieDescriptor` (tied embeddings break FSDP meta-tensor init / OOM on 30B-A3B).
- `_patch_unfuse_qwen3_omni_thinker_experts`: hooks `peft.get_peft_model`; `_convert_model_experts` swaps fused `Qwen3OmniMoeThinkerTextExperts` (3D `gate_up_proj`/`down_proj`) for `_Qwen3OmniMoeThinkerTextExpertsUnfused` per-expert `nn.Linear` modules (top-k routed via one-hot mask + `index_add_`); also disables PEFT's tf5 `gate_proj/up_proj → gate_up_proj` remap for this model (`_patched_get_mapping`/`_patched_convert`) and splits verl's comma-separated `target_modules` string into a set.
- `patch_hf_processor_for_qwen3_omni`: wraps `verl.utils.tokenizer.hf_processor` with a fallback that builds a `Qwen3OmniMoeProcessor`, attaches `thinker_config`, `spatial_merge_size`, `get_rope_index` (cast to int64 to avoid BF16 large-int rounding in FSDP), `get_llm_pos_ids_for_vision`, and a `_dedup_pad_tokens` helper; refreshes stale re-exports in `sys.modules` (`verl.utils`, `verl.workers.config.model`).
- `patch_hf_tokenizer_for_qwen3_omni`: wraps `verl.utils.tokenizer.hf_tokenizer` to load `chat_template` from `chat_template.json` when missing; patches already-imported `verl.*` module bindings.
- `patch_register_vllm_moe_model_weight_loader`: appends vllm-omni's `Qwen3OmniMoeThinkerForConditionalGeneration` to `verl.utils.vllm.patch.SUPPORTED_MOE_MODELS` so `patch_vllm_moe_model_weight_loader` re-attaches FusedMoE `weight_loader` (needed for vllm-ascend IPC weight sync).

## Flow
1. Trainer loads this module as `external_lib` → import-time `apply_qwen3_omni_thinker_patches()` runs all five patches.
2. Model build: verl resolves the architecture via the patched `_architecture_to_auto_class` → `AutoModelForMultimodalLM`; FSDP engine reads `_no_split_modules` and drops `_verl_strip_modules`; `tie_word_embeddings` reads False.
3. Data path: `hf_processor` fallback builds the Qwen3-Omni processor (rope index / pad-token dedup); `hf_tokenizer` fallback loads the chat template.
4. LoRA attach: `peft.get_peft_model` hook unfuses MoE experts, fixes `target_modules`, then delegates to the original.
5. Rollout weight sync: vLLM MoE weight-loader patch whitelist includes the Thinker class, so `weight_loader` attrs survive `process_weights_after_loading`.

## Integration
- Consumed by: legacy (pre-V1) trainer workers as a verl `external_lib` import target; referenced by `tests/gpu_smoke/select_gpu_smoke_groups.py`; V1 scripts (e.g. `run_qwen3_omni_thinker_gspo_lora_v1.sh`) are the migration path.
- Depends on: `transformers` (incl. `AutoModelForMultimodalLM`, `Qwen3OmniMoeConfig`, `Qwen3OmniMoeForConditionalGeneration`), `peft` (incl. `peft.utils.transformers_weight_conversion`), `torch`, `verl.utils.model`, `verl.utils.tokenizer`, `verl.utils.vllm.patch`, `vllm_omni.model_executor.models.qwen3_omni`.
- Key entry points: `apply_qwen3_omni_thinker_patches()` (applied on import), plus the individual `patch_hf_processor_for_qwen3_omni`, `patch_hf_tokenizer_for_qwen3_omni`, `patch_register_vllm_moe_model_weight_loader`.
