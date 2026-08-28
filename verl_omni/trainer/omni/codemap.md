# verl_omni/trainer/omni/

## Responsibility
The omni-modal (AR language/audio/video) trainer layer. Provides `OmniPPOTrainerSync` — a thin `PPOTrainerSync` subclass registered as `"omni_sync"` for verl's v1 PPO loop — and `OmniDirectPreferenceRayTrainer`, a standalone offline omni DPO trainer, plus `omni_algos.py` with the omni loss registry (`OmniDPOLoss`).

## Design
- **Delegation for online RL**: online policy-gradient omni training does not live here — `main_omni.run_omni` sets `trainer.use_v1=True` and delegates to verl's `run_ppo` with `TaskRunnerV1`; importing `verl_omni.trainer.omni` registers `@register_trainer("omni_sync")` so `OmniPPOTrainerSync` is picked up. Its only override is `_init_tokenizer`, sourcing tokenizer/processor from `OmniModelConfig` (`omega_conf_to_dataclass(..., OmniModelConfig)`) instead of plain HF loading.
- **`OmniDirectPreferenceRayTrainer`** (ray_omni_trainer.py): self-contained offline trainer requiring `algorithm.sample_source=offline`, `model_type=omni_model`, and `omni_loss.loss_mode=dpo`. Supports ref-in-actor (LoRA base as reference via `no_lora_adapter`) or an external `Role.RefPolicy` worker; `NoOpCheckpointManager` (no rollout weight sync needed); rollout.n or `paired_preference` expands batch sizes (`_expanded_preference_batch_size`); shuffle disabled for paired preference.
- **Loss lookup**: `omni_algos.get_omni_loss_fn(loss_mode)` returns `OmniDPOLoss` (with `OmniLossResult` carrying metrics like reward accuracy/margin); the same loss fn runs in workers during `_update_actor` and on the driver during `_validate`.

## Flow (OmniDirectPreferenceRayTrainer.fit — offline, no generation)
1. `Tracking` logger init; `_load_checkpoint()` (actor ckpt + dataloader state); `checkpoint_manager.update_weights` (no-op).
2. Optional pre-train `_validate`; then per batch: `_batch_dict_to_dataproto` (TensorDict + `multi_modal_inputs` as non-tensor) with `_omni_dpo_meta_info` (pad mode, micro-batch, temperature, global batch size).
3. Reward scores come pre-computed from the offline dataset (`batch.batch["sample_level_scores"]` → `sample_level_rewards`); no rollout/reward phase.
4. `_infer_reference_policy` → `ref_log_prob` (actor `infer_actor_batch` with `no_lora_adapter`, or `ref_policy_wg.infer_ref_batch`), unioned into the batch.
5. `_update_actor` → `actor_rollout_wg.update_actor(batch_td)` (DPO loss computed inside the worker with chosen/rejected pairs kept adjacent).
6. Checkpoint save at `save_freq` (with ESI-expiration force save), periodic `_validate` computing `dpo_loss`, `reward_accuracy`, `reward_margin`, `chosen/rejected_rewards` grouped by modality via `GroupedMetricMean`; metrics via `compute_data_metrics_diffusion`/timing/throughput helpers.

## Differences vs diffusion/v1
- No TransferQueue, replay buffer, agent loop, LLM servers, or rollout weight sync — offline preference data replaces online generation; DPO advantage is implicit in the pair loss.
- Online omni RL reuses verl v1 PPO (`PPOTrainerSync`/`TaskRunnerV1`) with `rollout.name: vllm_omni`, not this package's loop.
- Modality-aware metrics (`get_batch_modality`) and omni-specific tokenizer/processor loading via `OmniModelConfig` adapters.

## Integration
- Consumed by: `verl_omni/trainer/main_omni.py` — `get_ray_trainer_cls` returns `OmniDirectPreferenceRayTrainer` for `trainer_type=direct_preference` + `model_type=omni_model`; `RayTrainerTaskRunner` builds roles/pools/datasets and calls `init_workers()`/`fit()`; module import registers `omni_sync`.
- Depends on: verl v1 `PPOTrainerSync`/`register_trainer`, `verl_omni.workers.config` (`OmniModelConfig`), `verl_omni.utils.dataset.offline_mllm_dpo_dataset.get_batch_modality`, `verl_omni.utils.metrics_utils.GroupedMetricMean`, diffusion metric utils, `omni_algos.get_omni_loss_fn`.
- Key entry points: `OmniPPOTrainerSync`, `OmniDirectPreferenceRayTrainer`, `get_omni_loss_fn`, `OmniDPOLoss`, `OmniLossResult`.
