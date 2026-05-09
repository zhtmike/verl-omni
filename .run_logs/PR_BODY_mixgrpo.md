# PR Title

[algo, trainer, rollout] feat: integrate MixGRPO sliding-window SDE training

# PR Body

## Summary

Add trainer-side **MixGRPO** (sliding-window ODE-SDE rollout) support, implementing the algorithm from [arXiv:2507.21802](https://arxiv.org/abs/2507.21802).

- **SDE-window scheduler** (`sde_window_scheduler.py`): a training-time state machine that slides the SDE window across the denoising trajectory. Supports `random` (seeded uniform draw each step) and `progressive` (advance by `sde_window_size` every `iters_per_group` steps) strategies.
- **Algo config** (`DiffusionRolloutAlgoConfig`): new `algo_type` selector (`flow_grpo` / `mix_grpo`) plus MixGRPO-specific knobs (`sample_strategy`, `iters_per_group`, `seed`). Fully backward-compatible: `algo_type=flow_grpo` is the default.
- **Trainer integration**: `RayFlowGRPOTrainer.fit()` queries the scheduler each step and injects `sde_window_size` / `sde_window_range` overrides via `meta_info["algo_overrides"]`. Also injects a deterministic per-step `rollout_seed` derived from `data.seed + global_steps` for reproducible A/B comparisons.
- **Agent loop**: `DiffusionAgentLoopWorker` merges `algo_overrides` into rollout sampling params before dispatch. Derives per-rollout seeds from the base step seed so `rollout.n` group members see different initial noise (avoids advantage collapse).
- **LLMServerClient adapter** (prerequisite): adapts `verl-omni` to the upstream `verl` `LLMServerClient` refactor (verl-project/verl#6129), bumps `verl` pin to `a4351480`.

### Breaking changes

- Requires `verl >= a4351480` (the `LLMServerClient` API). The `verl` pin in `docs/start/install.md` and CI workflows is updated accordingly.
- `DiffusionAgentLoopWorker.__init__` signature changed: accepts `llm_client: LLMServerClient` instead of `servers` + `load_balancer_handle`.

## New files

| File | Description |
|---|---|
| `verl_omni/trainer/diffusion/sde_window_scheduler.py` | FlowGRPO / MixGRPO window schedulers |
| `docs/algo/mixgrpo.md` | Algorithm documentation and config reference |
| `examples/flowgrpo_trainer/run_qwen_image_ocr_lora_mixgrpo.sh` | MixGRPO recipe (progressive, 50-step inference) |
| `examples/flowgrpo_trainer/run_qwen_image_ocr_lora_mixgrpo_ab1.sh` | Matched-compute A/B-1 recipe (random, same knobs as FlowGRPO baseline) |
| `tests/schedulers/diffusion/test_sde_window_scheduler_on_cpu.py` | Unit tests for all scheduler variants |
| `tests/schedulers/diffusion/test_sde_window_scheduler_parity_on_cpu.py` | Parity tests: FlowGRPO scheduler == static baseline |
| `tests/pipelines/test_flow_match_sde_on_cpu.py` | SDE pipeline integration tests |
| `tests/pipelines/test_flow_match_sde_parity_on_cpu.py` | SDE pipeline parity tests |
| `tests/agent_loop/test_diffusion_agent_loop_seed_on_cpu.py` | Per-rollout seed derivation tests |

## Test plan

- [x] CPU unit tests pass for scheduler, pipeline, seed derivation, and config
- [x] End-to-end MixGRPO training (120 steps, 4×GPU, LoRA, OCR task) completes without error
- [x] Validation reward reaches 0.958 at step 120 (comparable to FlowGRPO baseline 0.964)
- [x] Training metrics (pg_clipfrac, ppo_kl, grad_norm) remain in the stable regime throughout
- [x] FlowGRPO baseline continues to work unchanged (backward compatibility verified)

## End-to-end results (wandb: `flow_grpo` project)

| Run | algo_type | Val reward @120 | pg_clipfrac (mean) | |ppo_kl| (mean) | grad_norm (max) |
|---|---|---|---|---|---|
| FlowGRPO R4 | flow_grpo | 0.964 | 0.010 | 2.4e-6 | 0.005 |
| FlowGRPO R5 | flow_grpo | 0.942 | 0.085 | 1.8e-4 | 0.131 |
| **MixGRPO ab1** | **mix_grpo** | **0.958** | **0.030** | **6.6e-6** | **0.006** |

AI assistance was used in developing this PR.
