# verl_omni/trainer/diffusion/

## Responsibility
The legacy (v0) diffusion RL training layer: Ray single-controller trainers for diffusion/flow models (`PolicyGradientRayTrainer`, `DirectPreferenceRayTrainer`), the diffusion algorithm math library (losses and advantage estimators such as Flow-GRPO, Flow-DPPO, DPO, NFT, KL, distillation), rollout-correction utilities, teacher management for distillation, and diffusion-specific metric computation. The v1 loop lives in the `v1/` subpackage (see its own map).

## Design
- **Base + paradigm subclasses** (`ray_diffusion_trainer.py`): abstract `BaseRayDiffusionTrainer` owns worker init (`init_workers`), colocated worker groups via `create_colocated_worker_cls`, dataloaders, validation (`_validate`), checkpointing (`_save_checkpoint`/`_load_checkpoint`), and media dumping (`_dump_generations` for images/videos+audio). `PolicyGradientRayTrainer.fit` runs the online RL loop; `DirectPreferenceRayTrainer.fit` runs offline preference training. `validate_separate_config` enforces constraints for separate trainer/rollout topology.
- **Registries** (`diffusion_algos.py`): `@register_diffusion_loss` fills `DIFFUSION_LOSS_REGISTRY` with `DiffusionLossFn` subclasses — `FlowGRPOLoss` (`flow_grpo`, `dance_grpo`), `FlowDPPOLoss`, `GRPOGuardLoss`, `DPOLoss`, `DiffusionNFTLoss`, `KLLoss`, `DistillKLLoss`, `DistillFlowMatchingMSELoss` — looked up by `get_diffusion_loss_fn`. `DiffusionAdvantageEstimator` enum + `DIFFUSION_ADV_ESTIMATOR_REGISTRY` with `get_diffusion_adv_estimator_fn` (e.g. `FLOW_GRPO`).
- **Support modules**: `diffusion_trainer_utils.py` (`_to_diffusion_worker_tensordict`, `old_policy_decay`, `OLD_POLICY_DECAY_SCHEDULES`, `NoOpCheckpointManager`), `rollout_correction.py` (off-policy IS/clip correction, bypass mode), `teacher_manager.py` (`DiffusionTeacherManager` for standalone teacher workers), `diffusion_metric_utils.py` (`compute_data_metrics_diffusion`, timing/throughput/reward-extra metrics).

## Flow (PolicyGradientRayTrainer.fit, one step)
1. `gen_batch.repeat(rollout.n)` → `async_rollout_manager.generate_sequences(...)` (agent-loop rollout servers); then `checkpoint_manager.sleep_replicas()` when colocated.
2. Reward: agent reward loop scores streamed during generation, or colocated RM via `_compute_reward_colocate` (`reward_loop_manager.compute_rm_score`).
3. `_compute_old_log_prob` (actor `infer_actor_batch` → `old_log_probs`, `old_prev_sample_mean`), optional `_compute_ref_log_prob` (`ref_log_prob` via ref worker or `no_lora_adapter` actor path).
4. `compute_advantage(...)` — driver-side Flow-GRPO advantage over denoising timesteps (`sample_level_rewards` expanded per timestep, optional `norm_adv_by_std_in_grpo`/`global_std`).
5. `_update_actor` → `actor_rollout_wg.update_actor(batch_td)` with the registered loss fn; then optional rollout correction, checkpoint save, weight update, validation at `test_freq`.

## Integration
- Consumed by: `verl_omni/trainer/main_diffusion.py` (TaskRunner builds `role_worker_mapping`/`ResourcePoolManager` and calls `trainer.init_workers()`/`fit()`); `main_omni.py` reuses `PolicyGradientRayTrainer`/`DirectPreferenceRayTrainer` for non-v1 omni runs.
- Depends on: `verl_omni.workers.engine_workers.ActorRolloutRefWorker`, `verl_omni.reward_loop`, verl single-controller (`RayWorkerGroup`, `ResourcePoolManager`), `verl.trainer.ppo.utils.Role`, `verl_omni.trainer.config.DiffusionAlgoConfig`.
- Key entry points: `BaseRayDiffusionTrainer`, `PolicyGradientRayTrainer`, `DirectPreferenceRayTrainer`, `compute_advantage`, `get_diffusion_loss_fn`, `get_diffusion_adv_estimator_fn`, `validate_separate_config`.
- Sub-map: `v1/codemap.md` — the TransferQueue-based v1 trainers registered for `trainer.use_v1=true`.
