# verl_omni/trainer/

## Responsibility
Top level of the verl-omni training system. Hosts the three Hydra entry points (`main_diffusion.py`, `main_diffusion_v1.py`, `main_omni.py`), the Hydra/YAML config tree under `config/`, and the per-paradigm trainer packages (`diffusion/`, `diffusion/v1/`, `omni/`). A run starts at one of the `main_*` entry points, which validate config, init Ray, and launch a Ray `TaskRunner` actor that builds worker groups and drives the trainer's `fit()` loop.

## Design
- **Hydra entry + Ray TaskRunner pattern** (inherited from veRL): `@hydra.main` loads `config/<name>.yaml`, then `run_*(config)` inits Ray and calls `runner.run.remote(config)` on a single-CPU `TaskRunner` actor. The TaskRunner maps veRL `Role` enums (`Role.ActorRollout`, `Role.ActorRolloutRef`, `Role.RewardModel`, `Role.TeacherModel`) to ray-remote worker classes (`verl_omni.workers.engine_workers.ActorRolloutRefWorker`) and builds a `ResourcePoolManager` (`global_pool`, optional `reward_pool`, `teacher_pool`).
- **Trainer selection by config**: `algorithm.trainer_type` (`policy_gradient` | `direct_preference`) plus `algorithm.sample_source` (`online` | `offline`) and `trainer.use_v1` pick the trainer class (`_get_trainer_cls` in `main_diffusion.py`, `get_ray_trainer_cls` in `main_omni.py`, `get_diffusion_trainer_cls` registry in `diffusion/v1/trainer_base.py`).
- **v1 mode**: v1 trainers use verl's v1 stack (TransferQueue, `ReplayBuffer`, `LLMServerManager`, `CheckpointEngineManager`, `AgentLoopManager`) with hook-based subclasses registered via `@register_diffusion_trainer` / `@register_trainer`.
- Optional nsys profiling of the controller via Ray `runtime_env.nsight`; optional `ray.timeline` dump.

## Flow
1. CLI → `main_diffusion_v1.main` (or `main_omni.main`): `auto_set_device` → `OmegaConf.resolve` → `validate_config` → attention checks (`fallback_fa3_if_unavailable`, `validate_attention_consistency`).
2. `run_diffusion_v1` / `run_omni` / `run_diffusion`: `ray.init()` with merged runtime env; launch `DiffusionTaskRunnerV1` / `RayTrainerTaskRunner` / `TaskRunner` actor; `ray.get(runner.run.remote(config))`.
3. Inside `TaskRunner.run`: map roles to workers (`add_actor_rollout_worker`), build resource pools (`init_resource_pool_mgr`), resolve model dir, load tokenizer/processor, create RL datasets/samplers (`verl_omni.utils.dataset.rl_dataset`), instantiate the trainer class, `trainer.init_workers()` then `trainer.fit()`.
4. v1 path (`DiffusionTaskRunnerV1.run`): force-enables TransferQueue, `tq.init(config.transfer_queue)`, selects trainer by `config.trainer.v1.trainer_mode` (`sync` / `separate_async`), `trainer.init()`, creates the agent loop manager (`verl_omni.agent_loop.create_diffusion_agent_loop_manager`), `trainer.fit(agent_loop_manager)`, `tq.close()` in `finally`.

## Integration
- Consumed by: recipes and CLI (`python -m verl_omni.trainer.main_diffusion_v1 ...`, `main_omni`), which override config.
- Depends on: `verl_omni/trainer/config/` (Hydra tree), `verl_omni.workers` (workers, engine workers, checkpoint engines), `verl_omni.agent_loop`, `verl_omni.reward_loop`, `verl_omni.utils.*`, and upstream veRL single-controller / ppo v1 infrastructure.
- Key entry points: `main_diffusion_v1.main` / `run_diffusion_v1`, `main_diffusion.main` / `run_diffusion`, `main_omni.main` / `run_omni`.

## Sub-maps
| Path | Focus |
|---|---|
| `config/codemap.md` | Hydra/YAML config tree, config groups, defaults composition |
| `diffusion/codemap.md` | Legacy (v0) diffusion Ray trainers + algorithm math (losses, advantage estimators) |
| `diffusion/v1/codemap.md` | v1 diffusion RL loop: TransferQueue, replay buffer, sync / separate-async modes |
| `omni/codemap.md` | Omni-modal trainer: verl v1 PPO delegation + offline omni DPO |
