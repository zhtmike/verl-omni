# verl_omni/pipelines/ltx2_flow_grpo/

## Responsibility
Flow-GRPO pipeline for **LTX-2/2.3** joint audio-video generation (`LTX2Pipeline` architecture, `flow_grpo` algorithm). The only GRPO pipeline with a custom **agent loop** (`LTX2DiffusionSingleTurnAgentLoop`) for multimodal prompt handling, and the only one doing audio+video (I2AV) latent joint denoising with phase-based recipes. Drives `trainer/main_diffusion.py` (typically via `recipe/flowgrpo_trainer/`).

## Design
- `diffusers_training_adapter.py`: `@DiffusionModelBase.register("LTX2Pipeline", algorithm="flow_grpo")` class `LTX23FlowGRPO(DiffusionModelBase)` — SDE scheduler via `FlowMatchSDEDiscreteScheduler`; `prepare_model_inputs` builds LTX-2 transformer inputs; `_predict` splits transformer output into video/audio predictions; `forward_and_sample_previous_step` performs the reverse-sample log-prob step.
- `agent_loop.py`: `@register("ltx2_diffusion_single_turn_agent")` class `LTX2DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop)` (from `verl.experimental.agent_loop.agent_loop`) — overrides `ct_build_initial_tokens` and `apply_chat_template` to turn multimodal `messages` into LTX-2 prompt text/token inputs.
- `vllm_omni_rollout_adapter.py`: class `LTX23PipelineWithLogProb(LTX2Pipeline)` on the vllm-omni LTX-2 stack — uses `LTXDenoiseExecutor`, `LTXVideoAudioStepAdapter`, `LTXPhaseRecipe`, `LTXAVState`, `LTXPromptContext`, `LTXRequestInputs` to run the multi-phase audio-video denoise loop while recording per-step trajectory log-probs into `DiffusionOutput.trajectory_*`.
- `common.py`: LTX-2 timestep/scheduler helpers (LTX uses its own sigma/shift conventions vs Qwen-Image's `calculate_shift`).

## Flow
1. Data: dataset → `LTX2DiffusionSingleTurnAgentLoop.apply_chat_template` builds the request; rollout server instantiates `LTX23PipelineWithLogProb`.
2. Rollout: phase recipes drive video+audio denoising (`LTXDenoiseExecutor` over `LTXAVState`); SDE noise injection records latents/timesteps/log-probs per step.
3. Training: `LTX23FlowGRPO.set_timesteps` mirrors the rollout schedule; `prepare_model_inputs` slices the trajectory per step; `forward_and_sample_previous_step` produces `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)` for the flow-GRPO loss in `trainer/diffusion/diffusion_algos.py`; rewards scored on generated video/audio via the reward loop.

## Integration
- Consumed by: diffusion trainer/engine via `(LTX2Pipeline, flow_grpo)` registry keys; agent-loop registry via the `"ltx2_diffusion_single_turn_agent"` name.
- Depends on: `pipelines/schedulers`, `pipelines.model_base`, vllm-omni `ltx2` module (`pipeline_ltx2`, `ltx2_denoise`, `ltx2_latents`, `ltx2_recipes`).
- Key entry points: `LTX23FlowGRPO`, `LTX23PipelineWithLogProb`, `LTX2DiffusionSingleTurnAgentLoop`.
- Sibling differences: only flow-GRPO pipeline with a registered agent loop and only one targeting audio-video generation; training adapter named `LTX23FlowGRPO` (not the model name) and handles dual video/audio prediction splits.
