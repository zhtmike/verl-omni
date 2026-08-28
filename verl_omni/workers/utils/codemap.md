# verl_omni/workers/utils/

## Responsibility
Worker-side helpers shared by engines and `engine_workers.py`: the loss entry points that bridge actor configs to the trainer algorithm registries (`losses.py`) and diffusion prompt-embed padding conversion (`padding.py`).

## Design
- `losses.py`:
  - `diffusion_loss(config: DiffusionActorConfig, model_output, data, dp_group=None)` — top-level diffusion loss dispatcher: resolves `get_diffusion_loss_fn(config.diffusion.loss_mode)` from `verl_omni.trainer.diffusion.diffusion_algos`, validates inputs, computes the loss result (`loss`, `metrics` as `Metric` objects); optionally adds KL (`use_kl_loss` × `kl_loss_coef`) and distillation terms (`use_distill_loss` × `distill_loss_coef` via `distill_loss_mode`); scales by `gradient_accumulation_steps` and `sp_size`.
  - `_apply_bypass_rc(log_prob, old_log_prob, rc_cfg, data, metrics)` — Rollout-Correction bypass mode: per-step IS/RS via verl's `compute_rollout_correction_and_rejection_mask` (current policy vs rollout policy log-probs), stashes `rollout_is_weights` into `data` (only `loss_type="ppo_clip"` supported); consumed by the engine's `forward_step` when `micro_batch["rollout_is_weights"]` exists.
  - `omni_loss(config: OmniActorConfig, model_output, data, dp_group=None)` — omni AR direct-preference loss dispatcher via `get_omni_loss_fn(config.omni_loss.loss_mode)` from `verl_omni.trainer.omni.omni_algos`, with the same grad-accum/`sp_size` scaling.
- `padding.py`: `embeds_padding_2_no_padding(data: TensorDict)` — converts padded `(bs, seq, dim)` `prompt_embeds`/`negative_prompt_embeds` (+ masks, expected left-aligned `[1111000...]`) into jagged `torch.nested` tensors by stripping per-sample padding; used in diffusion data prep so engines can unpad/pad for Ulysses SP later.
- `__init__.py`: license header only (no re-exports; consumers import submodules directly).

## Flow
1. `ActorRolloutRefWorker.init_model` builds `self.loss_fn = partial(diffusion_loss, config=actor_config)` (diffusion models) or `partial(omni_loss, config=actor_config)` (omni `direct_preference`); verl's `ppo_loss`/`distillation_ppo_loss` cover other paths.
2. `TrainingWorker.set_loss_fn` distributes the partial; engine `forward_step` invokes it per timestep/step with `model_output` + a per-step `data` TensorDict (`old_log_probs`, `advantages`, optional `ref_log_prob`, `ref/teacher/old_prev_sample_mean`, `rollout_is_weights`).
3. Diffusion bypass-RC: if `config.rollout_correction.bypass_mode` and the loss needs `log_probs`, `_apply_bypass_rc` runs before loss dispatch and writes weights/metrics.
4. Returned `(loss, metrics)` are aggregated per micro-batch in the engine's `postprocess_batch_func` and reduced in `TrainingWorker._postprocess_output`.

## Integration
- Consumed by: `verl_omni/workers/engine_workers.py` (loss_fn construction); engines call the injected loss fn, not this module directly; `embeds_padding_2_no_padding` is used by diffusion data preprocessing (pipelines/trainer side).
- Depends on: `verl_omni.trainer.diffusion.diffusion_algos.get_diffusion_loss_fn`, `verl_omni.trainer.omni.omni_algos.get_omni_loss_fn`, `verl.trainer.ppo.rollout_corr_helper.compute_rollout_correction_and_rejection_mask`, `verl.utils.metric.Metric`, `verl_omni.workers.config` (`DiffusionActorConfig`, `OmniActorConfig`).
- Key entry points: `diffusion_loss`, `omni_loss`, `embeds_padding_2_no_padding`.
