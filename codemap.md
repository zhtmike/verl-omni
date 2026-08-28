# Repository Atlas: verl-omni

## Project Responsibility

verl-omni is a multimodal/diffusion RL post-training framework built on [veRL](https://github.com/volcengine/verl) (RL training loops), veOmni (training engine), and vLLM-Omni (rollout inference). It post-trains image, video, and omni-modal generation models (Qwen-Image, SD3, Wan2.2, LTX-2, BAGEL, Qwen3-Omni, MiniMax H3, ...) with policy-gradient algorithms (flow-GRPO, DanceGRPO, MixGRPO, PPO), preference algorithms (DPO), and negative fine-tuning (NFT).

## System Entry Points

- `pyproject.toml` / `setup.py` — package metadata, extras (`gpu`, `train`, `dev`), vLLM-Omni pin.
- `verl_omni/trainer/main_diffusion_v1.py` — Hydra entry: v1 diffusion RL trainer (sync / separate-async).
- `verl_omni/trainer/main_diffusion.py` — Hydra entry: legacy v0 diffusion trainer (policy-gradient / direct-preference).
- `verl_omni/trainer/main_omni.py` — Hydra entry: omni-modal trainer (PPO / offline DPO).
- `verl_omni/pipelines/` — model+algorithm registration: pipelines register `DiffusionModelBase` / `VllmOmniPipelineBase` / `OmniModelBase` / `OmniRolloutPipelineBase` implementations that the trainer entry points resolve by config key.

## Architecture at a Glance

One training step (diffusion v1): rollout generation (vLLM-Omni via `agent_loop` + `workers/rollout`) → reward scoring (`reward_loop` + `utils/reward_score`) → old/ref log-prob & Flow-GRPO advantage (`trainer/diffusion/v1`) → policy update on an FSDP/veOmni engine (`workers/engine`) → weight sync back to rollout. Pipelines adapt each model family to both sides (diffusers training adapter + vLLM rollout adapter); `models/` holds cross-cutting monkey-patches; `trainer/config` is the Hydra config tree driving all of it.

## Directory Map (Aggregated)

| Directory | Responsibility Summary | Detailed Map |
|-----------|------------------------|--------------|
| `verl_omni/agent_loop/` | Rollout-side agent-loop workers: `DiffusionAgentLoopWorker`, composite dual AR+DiT reward loop, TransferQueue variant, single-turn registry. | [View Map](verl_omni/agent_loop/codemap.md) |
| `verl_omni/models/` | Monkey-patch adapter layer bridging diffusers/transformers model classes to the engines (FA3 varlen fix, Qwen-Image Ulysses fix; transformers adapters deprecated for v0.3.0). | [View Map](verl_omni/models/codemap.md) |
| `verl_omni/pipelines/` | Model+algorithm wiring: dual registry of training/rollout adapters; 15 pipelines (flow-GRPO, DanceGRPO, MixGRPO, DPO, NFT, omni) + SDE schedulers. | [View Map](verl_omni/pipelines/codemap.md) |
| `verl_omni/reward_loop/` | `OmniRewardLoopManager` (profiling fan-out) + `VisualRewardManager` / `MultiVisualRewardManager` (weighted multi-scorer aggregation). | [View Map](verl_omni/reward_loop/codemap.md) |
| `verl_omni/trainer/` | Hydra entry points, Ray TaskRunner pattern, and the three trainer families: diffusion legacy, diffusion v1 (sync/async), omni (PPO/DPO). | [View Map](verl_omni/trainer/codemap.md) |
| `verl_omni/utils/` | RLHF datasets (incl. DPO offline + visual reflection), reward scorer inventory (`utils/reward_score`), MFU counters, vLLM-Omni LoRA hijack sync. | [View Map](verl_omni/utils/codemap.md) |
| `verl_omni/workers/` | Ray workers: `ActorRolloutRefWorker`/`TrainingWorker`, FSDP + veOmni engine wrappers, vLLM-Omni async rollout server, worker config dataclasses. | [View Map](verl_omni/workers/codemap.md) |
| `verl_omni/version/` | Package version. | — (see `verl_omni/codemap.md`) |

Not mapped (by policy): `docs/` (contributing + user guides), `examples/` (per-pipeline run scripts + custom scorers), `scripts/`, `tests/` (CPU tests `test_*_on_cpu.py`), `docker/`, `reviews/`.

## Package-Level Atlas

For the in-package module table, see [verl_omni/codemap.md](verl_omni/codemap.md); every subpackage directory also carries its own `codemap.md`.
