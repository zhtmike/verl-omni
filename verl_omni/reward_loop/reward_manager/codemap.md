# verl_omni/reward_loop/reward_manager/

## Responsibility
Concrete reward managers for visual (image/video) diffusion RL. Each `run_single` scores exactly one sample's generated response against its ground truth and returns `{"reward_score", "reward_extra_info"}` for consumption by the reward loop / trainer.

## Design
- **`VisualRewardManager`** (`visual.py`): extends `verl.experimental.reward_loop.reward_manager.base.RewardManagerBase`.
  - Constructor falls back to `verl_omni.utils.reward_score.default_compute_score_image` when `compute_score` is `None` or the upstream default; detects sync vs async scoring functions via `inspect.iscoroutinefunction`; stores `reward_router_address` and `reward_model_tokenizer` for reward-model (RM) backed rewards.
  - `_validate_visual_response(response_visual, config, is_validate)`: enforces response dtype per `actor_rollout_ref.rollout.pipeline.output_type` — floating-point tensors for `"latent"`, `torch.uint8` pixels otherwise (selects `val_kwargs.pipeline` during validation).
  - `assemble_rm_scores(data, scores)` classmethod: packs per-sample image rewards into a `(batch_size, 1)` float32 tensor.
- **`MultiVisualRewardManager`** (`multi.py`): subclasses `VisualRewardManager` to aggregate N reward functions by weighted sum.
  - Reads `config.reward.reward_functions`: each entry has `path`/`name` (loaded via `verl.utils.import_utils.load_extern_object`), `weight` (default 1.0), `required` (bool or "true"/"false" string), and any extra config keys passed to the function. Validates total weight > 0.
  - `_filter_kwargs(all_kwargs, sig)` passes only parameters declared in each function's signature (everything if the function accepts `**kwargs`); `_multi_reward_placeholder` is a never-called sentinel passed to the parent constructor.
  - All sub-rewards sharing an RM use a single `reward_router_address` (one RM instance; multiple RMs unsupported by design, per docstring).

## Flow
1. `run_single(data)` asserts a single item; reads `batch["responses"]`, `non_tensor_batch["data_source"]`, `reward_model.ground_truth`, `extra_info`, `tool_extra_fields` (merged into `extra_info`), `__num_turns__`, and `reward_scores` (rollout-time scores added to `extra_info`).
2. Single manager: runs `compute_score(data_source, solution_image, ground_truth, extra_info, **extra_reward_kwargs)` — awaited directly if async, else via `loop.run_in_executor`. A dict result's `score` becomes the reward and remaining keys populate `reward_extra_info`; a bare number becomes the reward with `reward_extra_info["acc"] = score`.
3. Multi manager: for each sub-reward, merges global + per-reward extra args, filters to the signature, executes (await or executor). Success contributes `weight * score`; dict results also log `reward/{key}/{field}` extras. Failures raise if `required`, else log, set `reward/{key}/errors = 1`, and contribute 0. Final extras include per-key scores under `reward/{key}` and `reward/combined`.

## Integration
- Consumed by: `verl_omni.reward_loop` (re-exports `VisualRewardManager`), reward loop workers invoked via `compute_score.remote(...)` from agent-loop workers (`verl_omni.agent_loop.DiffusionAgentLoopWorker._compute_score`, `CompositeAgentLoopWorker._compute_score`).
- Depends on: `verl` (`DataProto`, `RewardManagerBase`, `load_extern_object`, upstream `default_compute_score`), `verl_omni.utils.reward_score.default_compute_score_image`.
- Key entry points: `VisualRewardManager.run_single`, `VisualRewardManager.assemble_rm_scores`, `MultiVisualRewardManager.run_single`, `_validate_visual_response`.
