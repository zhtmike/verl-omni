# verl_omni/trainer/diffusion/v1/

## Responsibility
The v1 diffusion RL training loop: TransferQueue-based, replay-buffer-driven policy-gradient trainers for diffusion models. Provides the abstract base `PolicyGradientDiffusionTrainerV1` with the full per-step compute pipeline, plus two registered mode implementations (`sync`, `separate_async`) differing only in rollout/weight-sync lifecycle hooks, and TQ<->DataProto conversion utilities.

## Design
- **Registry**: `DIFFUSION_TRAINER_REGISTRY` + `@register_diffusion_trainer(name)`; `get_diffusion_trainer_cls(config.trainer.v1.trainer_mode)` selects the class in `DiffusionTaskRunnerV1.run` (`main_diffusion_v1.py`).
- **`PolicyGradientDiffusionTrainerV1` (trainer_base.py, ABC)** mirrors upstream verl v1 `PPOTrainer` lifecycle (`init` → `_setup`/`on_init_end`, `fit`, `step`, hook methods `on_train_begin/end`, `on_step_begin/end`, `on_sample_begin/end`, `on_validate_begin/end`) while keeping the diffusion `DataProto` compute contract.
- **Infrastructure wired in `_setup`/`_init_online_rollout_stack`**: colocated `ActorRolloutRefWorker` worker group (`_init_colocated_workers` via `create_colocated_worker_cls`), `OmniRewardLoopManager`, verl `LLMServerManager`, `CheckpointEngineManager`, `StatefulDataLoader` train/val loaders, and a `ReplayBuffer`/`ReplayBufferAsync` (chosen by trainer_mode) with off-policy thresholds and `refill_fn=_add_prompts_to_generate`.
- **Mode subclasses**:
  - `PolicyGradientDiffusionTrainerV1Sync` (trainer_sync.py, `@register_diffusion_trainer("sync")`): colocated, no partial rollout; `on_init_end`/`on_step_end` → `checkpoint_manager.update_weights`, `on_sample_end` → `sleep_replicas`.
  - `PolicyGradientDiffusionTrainerV1SeparateAsync` (trainer_separate_async.py, `"separate_async"`): hybrid engine with colocated (naive ckpt backend) + standalone rollout stacks (`LLMServerManager` + `OmniCheckpointEngineManager` with nccl/nixl/mooncake); `HybridEngineMode` TRAINER/ROLLOUT switching, warmup batches, `sync_compatible` pause/resume of standalone generation, `DiffusionWholeSampleRetryLLMServerClient`, `DiffusionDetachActorWorker` as the actor role, and CPU-snapshot old-log-prob computation (`save_model_to_cpu`/`restore_model_from_cpu`) so every local update in a `parameter_sync_step` cycle uses cycle-start weights.
- **tq_utils.py**: `diffusion_tq_batch_to_dataproto`, `put_dataproto_fields_to_tq`, `sort_diffusion_tq_keys` — bridge TransferQueue KV rows and diffusion `DataProto`.

## Flow (per training step)
1. `fit` loop: optional `_validate`, `on_step_begin` → `step(metrics, timing_raw)`.
2. `step`: `_add_batch_to_generate()` fetches a gen batch (`_next_train_batch`/`_fetch_one_gen_batch`), tags prompts via `tq.kv_batch_put`, and calls `agent_loop_manager.generate_sequences(batch)` (rollout generation on rollout servers). Optionally prefetches all `parameter_sync_step` local batches.
3. `_sample_training_batch(batch_size)`: `replay_buffer.sample(...)` from TransferQueue; with `drop_incomplete_groups`, failed groups are evicted (`tq.kv_clear`) and exactly refilled via `_add_prompts_to_generate`.
4. `_train_sampled_batch`: convert TQ→DataProto → optional colocated RM reward (`_compute_reward_colocate`, sleeping replicas around it) → `_balance_batch` (pad to DP/mini-batch divisor) → `_compute_old_log_prob` (or rollout-correction bypass) → optional `apply_rollout_correction_to_diffusion_batch` → optional `_compute_ref_log_prob` → `_compute_advantage` (`extract_reward` → `sample_level_scores` expanded to per-timestep rewards → `compute_advantage` Flow-GRPO with `norm_adv_by_std_in_grpo`/`global_std`) → `_update_actor` (`actor_rollout_wg.update_actor`) → write `old_log_probs`/`advantages`/`returns`/scores back to TQ (`put_dataproto_fields_to_tq`).
5. Hooks: mode `on_sample_end` (sleep / mode switch), `on_step_end` (weight sync to rollout replicas), checkpoint save at `save_freq`, validation at `test_freq`, metrics + rollout dump, `tq.kv_clear` of consumed keys.
6. Validation (`_validate`): dispatch val batches through `partition_id="val"` in TransferQueue, sample back, reward-score, dump images via `_dump_generations`/wandb.

## Integration
- Consumed by: `DiffusionTaskRunnerV1` in `verl_omni/trainer/main_diffusion_v1.py` (forces `transfer_queue.enable=True`, `tq.init`, `trainer.init()`, `init_agent_loop_manager()`, `trainer.fit(agent_loop_manager)`).
- Depends on: verl v1 infra (`ReplayBuffer`, `LLMServerManager`, `CheckpointEngineManager`, `AgentLoopManager`, `SkipManager`, `MetricsAggregator`), `transfer_queue` package, `verl_omni.trainer.diffusion` (math/metrics/rollout correction), `verl_omni.workers` (`ActorRolloutRefWorker`, `DiffusionDetachActorWorker`, `OmniCheckpointEngineManager`, `DiffusionWholeSampleRetryLLMServerClient`), `verl_omni.reward_loop.OmniRewardLoopManager`.
- Key entry points: `get_diffusion_trainer_cls`, `PolicyGradientDiffusionTrainerV1`, `PolicyGradientDiffusionTrainerV1Sync`, `PolicyGradientDiffusionTrainerV1SeparateAsync`, all re-exported from `v1/__init__.py`.
