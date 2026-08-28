# verl_omni/

## Responsibility
`verl_omni` is the top-level Python package of the verl-omni multimodal/diffusion RL post-training framework. Its `__init__.py` does two things: reads the package version from `version/version` into `__version__`, and eagerly imports the registration side-effect modules (`models`, `pipelines`, `reward_loop`, `trainer`, `workers.engine`, `workers.rollout`) so that all `@register` decorators and model patches fire at import time.

## Design
- **Import-for-registration root**: `verl_omni/__init__.py` is deliberately minimal — no class definitions, only version loading and `import verl_omni.X  # noqa: F401` statements that trigger auto-registration of pipelines, rollout modes, reward loops, and inference engines.
- **Layered subpackages**: model patches (`models`) → per-model rollout pipelines (`pipelines`) → rollout orchestration (`agent_loop`) → reward computation (`reward_loop`) → RL training entrypoints (`trainer`) → engine/rollout worker implementations (`workers`) → shared helpers (`utils`).

## Layout
| Subpackage | Responsibility | Codemap |
|---|---|---|
| [`agent_loop/`](agent_loop/codemap.md) | Diffusion/AR+DiT rollout agent loops: `DiffusionAgentLoopWorker`, `CompositeAgentLoopWorker`, `DiffusionSingleTurnAgentLoop`, TransferQueue-backed `DiffusionAgentLoopWorkerTQ`. | [codemap](agent_loop/codemap.md) |
| [`models/`](models/codemap.md) | Model patches for diffusers/transformers backends (e.g. `apply_flash_attention_3_varlen_hub_fix`, `apply_qwen_image_ulysses_mask_fix`), applied at import time. | — |
| [`pipelines/`](pipelines/codemap.md) | Per-model rollout pipelines (Qwen-Image, SD3, Wan2.2, Bagel, LTX-2, MiniMax-H3, Qwen3-Omni, etc.) × algorithms (GRPO/DPO/NFT/DanceGRPO); base classes `model_base.py`, `non_diffusers_model_base.py`, shared `request_batch.py`, `diffusion_rollout_output.py`, `schedulers/`. | — |
| [`reward_loop/`](reward_loop/codemap.md) | Reward orchestration: `OmniRewardLoopManager` (profiler control over reward-model replicas) and visual reward managers in `reward_manager/`. | [codemap](reward_loop/codemap.md) |
| [`trainer/`](trainer/codemap.md) | Training entrypoints (`main_diffusion.py`, `main_diffusion_v1.py`, `main_omni.py`) plus `diffusion/`, `omni/`, and `config/` packages. | — |
| [`utils/`](utils/codemap.md) | Shared helpers: config (`config.py`), dataset, diffusion attention, FSDP utilities, metrics/MFU, profiler, tracking, `reward_score/` (incl. `default_compute_score_image`), vLLM-Omni glue. | — |
| [`workers/`](workers/codemap.md) | Ray worker infrastructure: inference `engine/` implementations, `engine_workers.py`, `rollout/` replicas, `checkpoint_engine.py`, `detach_actor_worker.py`, worker `config/`. | — |
| `version/` | Single `version` text file (e.g. `0.2.0rc1`) read by `__init__.py` to set `verl_omni.__version__`. | — |

## Flow
1. `import verl_omni` → `__version__` loaded from `version/version`.
2. Imports of `models`, `pipelines`, `reward_loop`, `trainer`, `workers.engine`, `workers.rollout` register all model patches, rollout pipelines, reward loops, and engines into their respective verl registries.
3. A training run starts from `trainer/main_*.py`, which builds the Ray trainer, agent-loop workers (`agent_loop/`), rollout replicas (`workers/`), and reward workers (`reward_loop/`).

## Integration
- Consumed by: user entrypoints (`main_*.py` scripts / CLI), and the veRL framework via the registries populated at import.
- Depends on: `verl` (veRL core: `DataProto`, registries, `AgentLoopBase`, `RewardLoopManager`), `vllm-omni` (rollout serving), `torch`/`ray`/`hydra`.
- Key entry points: `verl_omni.__version__`; side-effect imports in `verl_omni/__init__.py`.
