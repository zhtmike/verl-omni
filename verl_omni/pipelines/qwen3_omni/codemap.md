# verl_omni/pipelines/qwen3_omni/

## Responsibility
Wires Qwen3-Omni-Moe (unified understanding + generation: thinker → talker → code2wav) into the omni-modal RL trainer path (`verl_omni/trainer/main_omni.py` → `trainer/omni`), unlike the diffusion pipelines which use `main_diffusion.py`. This directory provides only adapters: the rollout pipeline *topology* (delegated unchanged to vLLM-Omni) and the thinker-stage *training* model setup for verl's FSDP engine.

## Design
- `omni_rollout_adapter.py`: `Qwen3OmniRolloutAdapter` registered via `@OmniRolloutPipelineBase.register("qwen3_omni_moe")`. It maps three `pipeline_mode` values onto vLLM-Omni's frozen stage pipelines (`QWEN3_OMNI_PIPELINE`, `QWEN3_OMNI_THINKER_ONLY_PIPELINE`): `thinker_only` (stage 0, text), `thinker_talker` (stages 0-1, codec), `full` (stages 0-2, audio waveform). Classmethods: `build_stage_configs`, `rollout_flags` (e.g. stage 0 `return_hidden_states=True` for talker modes), `get_pipeline_id`, `ensure_pipeline_registered` (registers the thinker-only variant via `register_pipeline`), `get_engine_hf_overrides` (`enable_audio_output: False` for thinker-only), `get_stage_engine_extras` (`model_arch="Qwen3OmniMoeThinkerForConditionalGeneration"`).
- `thinker_training_adapter.py`: `Qwen3OmniThinkerAdapter` registered via `@OmniModelBase.register("Qwen3OmniMoeForConditionalGeneration", stage="thinker")`. `get_strip_modules` removes `talker`/`code2wav`/`code_predictor`; `configure_model` redirects `forward`, `get_input_embeddings`, `set_input_embeddings` to `module.thinker` before FSDP wrapping; `configure_processor` swaps `processor.config` to `thinker_config`, binds int64-cast `get_rope_index`, `get_llm_pos_ids_for_vision`, `get_rope_index_kwargs` (audio seqlens), and `dedup_pad_tokens` (collapses consecutive image/video/audio pad tokens); `configure_tokenizer` loads `chat_template.json`.

## Flow
1. Entry: `python3 -m verl_omni.trainer.main_omni` with `algorithm.trainer_type=direct_preference` (preference DPO, `examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh`) or policy-gradient (GRPO/GSPO, `examples/gspo_trainer/qwen3_omni/*.sh`).
2. Data: `QwenOmniRLHFDataset` (`verl_omni/utils/dataset/omni_rl_datasets.py`, extends `RLHFDataset`) with `qwen3_omni_transform.py` for multimodal preprocessing; offline preference training uses `OfflineMLLMDPODataset` (`verl_omni/utils/dataset/offline_mllm_dpo_dataset.py`).
3. Rollout: omni agent loop uses the bound processor helpers (RoPE index, pad-token dedup); the rollout engine instantiates the adapter's stage configs; thinker-only mode strips audio output.
4. Reward: e.g. VLM-as-judge (`examples/dpo_trainer/qwen3_omni/vlm_as_judge.py` / `eval_vlm_as_judge.sh`) or task-specific reward scores in `verl_omni/utils/reward_score/`; trainer updates the thinker only.

## Integration
- Consumed by: `verl_omni/pipelines/__init__.py`; `main_omni.py` → omni trainer (`trainer/omni`); `verl_omni/models/transformers/qwen3_omni_thinker.py` supplies the thinker training model.
- Depends on: `verl_omni.pipelines.model_base` (`OmniRolloutPipelineBase`, `OmniModelBase`), vLLM-Omni `vllm_omni.config.pipeline_registry.register_pipeline` and `vllm_omni.model_executor.models.qwen3_omni.pipeline`, transformers `Qwen3OmniMoeThinkerForConditionalGeneration`.
- Key entry points: `Qwen3OmniRolloutAdapter`, `Qwen3OmniThinkerAdapter`.
- Differences from diffusion pipelines: autoregressive token-level RL on the thinker (standard verl FSDP engine, log-prob based losses) instead of the diffusion trainer; multi-stage vLLM-Omni pipeline topology rather than a single diffusion pipeline.
