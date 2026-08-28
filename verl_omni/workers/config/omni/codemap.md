# verl_omni/workers/config/omni/

## Responsibility
Config dataclasses for omni (thinker/talker multimodal AR) model training: `OmniModelConfig` for model/tokenizer/processor loading and `OmniActorConfig` + `OmniLossConfig` for FSDP actor training (policy-gradient or direct-preference/DPO). Consumed by `OmniFSDPEngine` and the omni loss path.

## Design
- `model.py`: `OmniModelConfig(BaseConfig)` (`model_type="omni_model"`) — separate `path` / `hf_config_path` / `tokenizer_path`; `model_stage` ∈ {thinker, talker, all} selects the HF sub-config (`thinker_config`/`talker_config`, incl. `tie_word_embeddings` propagation); builds `hf_config` via `AutoConfig.from_pretrained` with `attn_implementation` from `override_config`; tokenizer/processor built through the model adapter (`OmniModelBase.get_class_by_name(architecture, model_stage, external_lib).configure_tokenizer/configure_processor`); training flags (`use_remove_padding`, gradient checkpointing, `use_liger`, `use_fused_kernels`, `tiled_mlp`, `mtp: MtpConfig`), LoRA fields incl. `lora_dtype`, `policy_state_adapters`, `fsdp_layer_prefixes`, multimodal token budgets (`max_image_tokens`/`max_audio_tokens`/`max_video_tokens`); `get_processor()` falls back to tokenizer.
- `actor.py`: `OmniLossConfig(BaseConfig)` — direct-preference (DPO) hyperparams (`loss_mode="dpo"`, `beta`, `label_smoothing`, `loss_type` ∈ {sigmoid, ipo}, `average_log_prob`, `refer_model_precision`); docstring explains the split vs. verl's `policy_loss` block used by `policy_gradient` trainers. `OmniActorConfig(FSDPActorConfig)` adds `trainer_type` ∈ {`direct_preference`, `policy_gradient`} and nested `omni_loss`, validated in `__post_init__`.

## Flow
1. `ActorRolloutRefWorker.init_model` converts `config.model` to `OmniModelConfig` and, for omni models, sets `actor_config.model_config.trainer_type` from the actor config.
2. Loss selection in `engine_workers`: `trainer_type == "direct_preference"` ⇒ `partial(omni_loss, config=actor_config)` (which dispatches via `get_omni_loss_fn(loss_mode)` from `verl_omni.trainer.omni.omni_algos`); `policy_gradient` ⇒ verl's `ppo_loss` path.
3. `OmniFSDPEngine._build_module` uses `hf_config`, `architecture`, `model_stage` (via `OmniModelBase.get_class_by_name`) and rejects `use_liger`/`use_fused_kernels`; AR-mode rollout uses `OmniModelConfig` in `vLLMOmniHttpServer._init_model_config`.

## Integration
- Consumed by: `verl_omni/workers/engine_workers.py`, `engine/fsdp/omni_impl.py` (`OmniFSDPEngine`), `rollout/vllm_rollout/vllm_omni_async_server.py` (AR mode), `utils/losses.py` (`omni_loss`).
- Depends on: `verl.base_config.BaseConfig`, `verl.workers.config.FSDPActorConfig`, `verl.workers.config.model.MtpConfig`, `transformers.AutoConfig`, `verl_omni.utils.fs.resolve_model_local_dir`, `verl_omni.pipelines.model_base.OmniModelBase`.
- Key entry points: `OmniModelConfig`, `OmniActorConfig`, `OmniLossConfig` (re-exported from `verl_omni.workers.config`).
