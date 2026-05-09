# PR draft

**Open at:** https://github.com/verl-project/verl-omni/compare/main...samithuang:verl-omni:feat/adapt-verl-llm-server-client?expand=1

**Title:**
```
[BREAKING][rollout] feat: adapt to verl LLMServerClient refactor
```

**Body:**

## Summary

Adapt verl-omni's diffusion agent loop and ray trainer to [verl-project/verl#6129](https://github.com/verl-project/verl/pull/6129), which removed `AsyncLLMServerManager` and made `AgentLoopManager` / `AgentLoopWorker` consume an `LLMServerClient` produced by a separately-owned `LLMServerManager`. Without this PR, the FlowGRPO trainer fails at startup against any verl commit that includes `#6129`:

```
ImportError: cannot import name 'AsyncLLMServerManager' from 'verl.experimental.agent_loop.agent_loop'
```

### verl-omni source changes

- `verl_omni/agent_loop/diffusion_agent_loop.py` — `DiffusionAgentLoopWorker.__init__` now takes `(config, llm_client, teacher_client, reward_loop_worker_handles)`, matching the positional contract that `AgentLoopManager.create()` uses when spawning workers in upstream verl. `_get_rollout_and_model_config` was also dropped upstream, so the config slicing is inlined to keep the diff minimal.
- `verl_omni/trainer/diffusion/ray_diffusion_trainer.py` — the trainer now creates an `LLMServerManager` first, hands its client to `AgentLoopManager.create()`, and uses `llm_server_manager.get_replicas()` (instead of `async_rollout_manager.rollout_replicas`) to wire `CheckpointEngineManager`. This mirrors the pattern in upstream `verl/trainer/ppo/ray_trainer.py`.
- `tests/agent_loop/test_diffusion_agent_loop.py` — updated for the new API; in standalone test mode `LLMServerManager` spins up its own replicas via `rollout.nnodes` / `n_gpus_per_node`.

### Pin / docs / CI

- Bump the pinned verl commit in `docs/start/install.md` and the three CI workflow files to `a4351480871347092436d17573ad3ccf75b24122` (the merge commit of [verl-project/verl#5209](https://github.com/verl-project/verl/pull/5209)). This is the first commit that ships `verl/experimental/reward_loop/router/` in the wheel **and** contains the `#6129` refactor that this PR adapts to. Once this PR lands the workaround in #51 is no longer required (#51 can be closed without merging).
- Restore the simple `uv pip install git+...@<commit>` install line in `docs/start/install.md` (removes the temporary editable-install workaround introduced in #51).

## BREAKING change

`DiffusionAgentLoopWorker.__init__` signature changed:

| before | after |
| --- | --- |
| `(config, servers, load_balancer_handle, teacher_servers=None, teacher_load_balancer_handle=None, reward_loop_worker_handles=None)` | `(config, llm_client, teacher_client=None, reward_loop_worker_handles=None)` |

Any downstream code that subclasses or directly instantiates `DiffusionAgentLoopWorker` must be updated. No public CLI / config / dataset surface is affected.

## Why this is not a duplicate PR

The corresponding upstream verl change (`#6129`) is merged. There is no other open verl-omni PR adapting to it (checked via `gh pr list --repo verl-project/verl-omni --search 'LLMServerClient'`). The only related open PR is #51 (router packaging workaround), which this PR makes obsolete.

## Test

End-to-end FlowGRPO LoRA training was run against the post-`#6129` verl pin, using `examples/flowgrpo_trainer/run_qwen_image_ocr_lora_local.sh` with `trainer.total_training_steps=450` (resuming from `global_step_301`):

```
training/global_step:302 - actor/loss:-0.00130 - actor/grad_norm:0.00316 - actor/lr:0.0003
critic/rewards/mean:0.9056 - critic/advantages/mean:-1.30e-08
timing_s/gen:228.2 - timing_s/reward:29.1 - timing_s/old_log_prob:32.0
timing_s/update_actor:109.3 - timing_s/update_weights:10.2 - timing_s/step:408.8
perf/throughput:0.313 samples/GPU/s   (baseline: ~0.305)

training/global_step:303
training/global_step:304   (run paused here for review)
```

This proves every code path the new API touches: `LLMServerManager.create()`, server replica launch, `AgentLoopManager.create(config=..., llm_client=...)`, `DiffusionAgentLoopWorker` Ray actor instantiation with the new positional signature, rollout via `LLMServerClient`, FSDP→vLLM-Omni weight sync via `llm_server_manager.get_replicas()`, reward server, PPO update, and wandb logging. Throughput matches the pre-refactor baseline.

A from-scratch end-to-end run was also verified (`trainer.resume_mode=disable`, fresh LoRA weights):

```
training/global_step:1 - actor/loss:1.03e-06 - actor/grad_norm:0.000354 - actor/lr:0.0003
critic/rewards/mean:0.6617 - critic/advantages/mean:-9.31e-09 - critic/rewards/std_mean:0.291
```

The lower reward mean (0.66 vs 0.91 in the resume run) and tiny loss/grad-norm match the expected behavior of step 1 of LoRA training from base weights, confirming `LLMServerManager` initializes the rollout servers cleanly when there is no prior checkpoint.

AI assistance was used to write this PR.
