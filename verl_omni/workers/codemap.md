# verl_omni/workers/

## Responsibility
The Ray actor worker layer of verl-omni: defines the long-lived worker processes that own a training engine (FSDP/FSDP2 or veOmni), a rollout engine (vLLM-Omni async server), and the shared config dataclasses that parameterize both. It adapts upstream verl's SPMD worker/`EngineRegistry` machinery to diffusion (`diffusion_model`, `diffusion_dpo_model`, `diffusion_nft_model`) and omni (`omni_model`) model types, and implements the trainer→rollout weight sync protocol (colocated ZMQ/IPC and disaggregated checkpoint-engine paths).

## Design
- **Hybrid ray worker**: `ActorRolloutRefWorker` (`engine_workers.py`) — one `verl.single_controller.base.Worker` that can host actor / rollout / ref / teacher roles, each role backed by an inner `TrainingWorker` (Tinker-like coarse API over a `BaseEngine` from verl's `EngineRegistry`).
- **TrainingWorker**: thin registered-method facade over the engine: `reset`, `train_mini_batch`, `train_batch`, `infer_batch`, `to`, `save_checkpoint`/`load_checkpoint`, `copy_adapter`/`ema_update_adapter`; per-mesh dispatch via `make_nd_compute_dataproto_dispatch_fn(mesh_name=...)`.
- **Engine plugin registry**: engines register themselves with `@EngineRegistry.register(model_type=..., backend=[...], device=[...])`; `TrainingWorker.__init__` resolves the engine with `EngineRegistry.new(...)`, optionally via `auto_select_engine_optim_fn`.
- **Detach/snapshot worker**: `DiffusionDetachActorWorker` (`detach_actor_worker.py`) mixes `verl.experimental.separation...DetachActorWorker` with the hybrid actor to keep CPU parameter snapshots (`save_model_to_cpu` / `restore_model_from_cpu`).
- **Checkpoint engine bridge**: `OmniCheckpointEngineManager` (`checkpoint_engine.py`) extends verl's `CheckpointEngineManager` to forward the actor LoRA `peft_config` to standalone rollout replicas via `collective_rpc("set_pending_lora_peft_config", ...)`.

## Flow
1. Trainer dispatches `init_model` on `ActorRolloutRefWorker`: builds `DiffusionModelConfig`/`OmniModelConfig`, then per role builds `TrainingWorker`s (ref with forced `MtpConfig(enable=False)`, actor with a loss fn from `verl_omni.workers.utils.losses` — `diffusion_loss`, `omni_loss`, `ppo_loss`, or `distillation_ppo_loss`), frozen distillation teachers (`build_teacher_training_config`), and the rollout via `get_rollout_class` + a `(dp, infer_tp, infer_pp)` device mesh.
2. Training: controller calls `update_actor` → `TrainingWorker.train_mini_batch` → splits TensorDict into mini-batches → `train_batch` → `engine.train_batch(data, loss_function)` → `optimizer_step` (grad clip + non-finite skip) → metrics gathered in `_postprocess_output` (DP all-reduce, MFU via `DiffusionFlopsCounter`/`FlopsCounter`).
3. Weight sync: `update_weights(global_steps, mode)` — `mode="naive"` gathers per-tensor params with `engine.get_per_tensor_param` and calls `rollout.update_weights`; LoRA-only steady state uses the fast path (`BucketedWeightSender` over a per-update ZMQ handle from `zmq_utils`). Non-naive modes push via `checkpoint_engine.send_weights`; `OmniCheckpointEngineManager` handles the standalone-replica LoRA peft_config handoff.
4. Lifecycle: rollout sleep/wake (`sleep_level=1`, `resume(tags=["weights"|"kv_cache"])`), actor param offload (`_offload_actor_and_empty_cache`), checkpoints via `save_checkpoint`/`load_checkpoint`.

## Integration
- Consumed by: `verl_omni.pipelines`/ray trainer controllers and verl's `RayWorkerGroup` dispatch machinery; standalone-async mode uses `OmniCheckpointEngineManager` + rollout replicas.
- Depends on: `verl.single_controller`, `verl.workers.engine` (`BaseEngine`, `EngineRegistry`), `verl.checkpoint_engine`, `verl_omni.workers.config`, `verl_omni.workers.utils.losses`, `verl_omni.utils.mfu`, `verl_omni.pipelines.utils`.
- Sub-maps:
  - `config/` — pydantic/dataclass configs (base, `diffusion/`, `omni/`)
  - `engine/` — training engine abstraction + `fsdp/`, `veomni/` implementations
  - `rollout/` — rollout base/replica + `vllm_rollout/`
  - `utils/` — loss entry points and padding helpers
- Key entry points: `ActorRolloutRefWorker.init_model`, `TrainingWorker.train_mini_batch`, `ActorRolloutRefWorker.update_weights`, `OmniCheckpointEngineManager.update_weights`.
