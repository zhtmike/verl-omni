# verl_omni/trainer/config/

## Responsibility
Hydra-style configuration tree for all trainers. Top-level YAMLs (`diffusion_trainer.yaml`, `omni_trainer.yaml`, `omni_megatron_trainer.yaml`) compose per-component config groups via Hydra `defaults` lists; nested directories hold the component YAMLs. `algorithm.py` defines the validated Python dataclass configs (`DiffusionAlgoConfig`, `OmniAlgoConfig`), and `_generated_*.yaml` files are CI-checked full expansions used for drift detection.

## Design
- **Defaults composition**: `diffusion_trainer.yaml` composes groups like `diffusion/actor@actor_rollout_ref.actor: ${diffusion/model_engine}_actor` — the engine choice (`dp_diffusion` vs `veomni_diffusion`) interpolates the actor/ref/model_engine/optim variants. `omni_trainer.yaml` extends verl's `ppo_trainer` (via `hydra.searchpath: pkg://verl.trainer.config`) with `omni/model@actor_rollout_ref.model: omni_model` and sets `rollout.name: vllm_omni`. `- _self_` last so the top file overrides groups.
- **Dataclass instantiation**: `_target_` fields (e.g. `verl_omni.workers.config.omni.OmniActorConfig`, `OmniLossConfig`) let `verl.utils.config.omega_conf_to_dataclass` build typed configs; interpolation (`trainer_type: ${algorithm.trainer_type}`) mirrors algorithm settings into the actor.
- **Python-side validation**: `DiffusionAlgoConfig` validates `adv_mode`, `old_policy_decay_schedule` (against `OLD_POLICY_DECAY_SCHEDULES`), `old_policy_decay`, `timestep_fraction`; `OmniAlgoConfig` validates `trainer_type` / `sample_source`. Both extend `verl.base_config.BaseConfig`.
- YAML format is CI-enforced (comments above each field, blank lines between fields, no inline comments).

## Config tree
| Path | Contents |
|---|---|
| `diffusion_trainer.yaml` | Root defaults for diffusion training (legacy + v1); `actor_rollout_ref.model.model_type: diffusion_model` |
| `omni_trainer.yaml` | Root defaults for omni AR training; extends verl `ppo_trainer`; `rollout.name: vllm_omni`, `actor.omni_loss` (loss_mode `dpo`, `beta`, `label_smoothing`, `loss_type`) |
| `omni_megatron_trainer.yaml` | Megatron-backend omni variant |
| `_generated_*.yaml` | Full config expansions (`diffusion_trainer`, `diffusion_veomni_trainer`, `omni_trainer`, `omni_megatron_trainer`) for CI format/drift checks |
| `algorithm.py` | `DiffusionAlgoConfig` (flow_grpo default, rollout correction), `OmniAlgoConfig` (offline DPO default) |
| `data/legacy_data.yaml` | Data group (train/val files, batch sizes, sampler, padding) |
| `diffusion/actor/` | `diffusion_actor.yaml`, `dp_diffusion_actor.yaml`, `veomni_diffusion_actor.yaml` — actor optimizer/loss config per engine |
| `diffusion/distillation/diffusion_distillation.yaml` | On-policy distillation (teacher pool) settings |
| `diffusion/engine/` | `diffusion_fsdp.yaml`, `diffusion_veomni.yaml` — engine backend selection |
| `diffusion/model/diffusion_model.yaml` | Model/pipeline config (path, architecture, `pipeline.height/width`, `vae_scale_factor`) |
| `diffusion/model_engine/` | `dp_diffusion.yaml`, `veomni_diffusion.yaml` — the `diffusion/model_engine` group driving `${...}_actor`/`${...}_ref` interpolation |
| `diffusion/ref/` | `diffusion_ref.yaml`, `dp_diffusion_ref.yaml`, `veomni_diffusion_ref.yaml` — reference policy config |
| `diffusion/rollout/diffusion_rollout.yaml` | Rollout config (n, agent loop, `checkpoint_engine`, seeds) |
| `omni/model/omni_model.yaml` | `OmniModelConfig` (`model_type: omni_model`, tokenizer/processor adapters, `policy_state_adapters`) |
| `optim/` | `fsdp.yaml`, `veomni_diffusion.yaml` — optimizer configs |
| `profiler/profiler.yaml` | `global_profiler` group (tool, steps, nsys options) |
| `reward/reward.yaml` | `reward` group incl. `reward_model` (enable, resource pool sizing) |

## Flow
1. Hydra loads the root YAML with `config_path="./config"` from the `main_*` entry points; `defaults` merge group files into one `DictConfig`.
2. `OmegaConf.resolve(config)` evaluates interpolations; `verl_omni.utils.config.validate_config` cross-checks the merged config.
3. `omega_conf_to_dataclass` instantiates `_target_` dataclasses inside workers/trainers (e.g. `checkpoint_engine`, `OmniModelConfig`, `DiffusionAlgoConfig` via `verl_omni.trainer.config` package exports).

## Integration
- Consumed by: `verl_omni/trainer/main_diffusion.py`, `main_diffusion_v1.py`, `main_omni.py` (hydra config_path), recipes overriding groups.
- Depends on: verl's `verl.trainer.config` searchpath (e.g. `ppo_trainer`), `verl.base_config.BaseConfig`, `verl_omni.workers.config` dataclasses, `diffusion_trainer_utils.OLD_POLICY_DECAY_SCHEDULES`.
- Key entry points: `verl_omni.trainer.config` package (`DiffusionAlgoConfig`, `OmniAlgoConfig`), root YAMLs selected by `@hydra.main(config_name=...)`.
