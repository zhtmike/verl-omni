# verl_omni/reward_loop/

## Responsibility
Reward orchestration layer for diffusion/omni RL training. Provides `OmniRewardLoopManager`, a veRL `RewardLoopManager` subclass that adds profiler start/stop control over the reward-model rollout replicas, and re-exports the visual reward managers from `reward_manager/` that actually compute per-sample scores.

## Design
- `OmniRewardLoopManager` (`reward_loop.py`): extends `verl.experimental.reward_loop.RewardLoopManager`. Adds:
  - `start_profile(**kwargs)` / `stop_profile()` — fan out `start_profile`/`stop_profile` to every rollout replica of the reward-model manager (`self.reward_model_manager.rollout_replicas`) via `_run_on_replicas`, which gathers `getattr(replica, method)(...)` with `asyncio.run`. Both are no-ops when no reward model is configured.
  - Rationale (per docstring): reward-model servers share the same `RolloutReplica` stack as actor rollout servers; upstream `RewardLoopManager` simply lacks a caller for the existing per-replica profiler methods. The trainer invokes these around the scoring phase (generation phase for streaming rewards, or `compute_rm_score` in colocate mode), configured via `reward.reward_model.rollout.profiler`.
- `reward_manager/`: `VisualRewardManager` and `MultiVisualRewardManager` — see [reward_manager/codemap.md](reward_manager/codemap.md).

## Flow
1. The trainer instantiates the reward loop manager with a reward-model manager (a pool of `RolloutReplica` serving the reward model).
2. During a training/validation phase where reward models score samples, the trainer calls `start_profile()` before and `stop_profile()` after; each call fans the profiler command to all replicas concurrently.
3. Actual scoring: agent-loop workers (or the manager, in colocate mode) dispatch `compute_score` to reward workers, which use `VisualRewardManager`/`MultiVisualRewardManager.run_single` to produce `{"reward_score", "reward_extra_info"}`.

## Integration
- Consumed by: `trainer/` (main diffusion/omni training loops) which wraps reward-model scoring phases with profiling.
- Depends on: `verl.experimental.reward_loop.RewardLoopManager`, `verl_omni.reward_loop.reward_manager` (`VisualRewardManager`), Ray rollout replicas from `verl_omni.workers.rollout`.
- Key entry points: `OmniRewardLoopManager.start_profile`, `OmniRewardLoopManager.stop_profile`, `_run_on_replicas`; re-exported `VisualRewardManager`.
