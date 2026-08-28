# verl_omni/agent_loop/

## Responsibility
Implements rollout agent loops for diffusion and composite (AR+DiT) models. A "worker" takes a batch of prompts (`DataProto`/`TensorDict`), fans each sample out to an async per-sample agent loop that talks to the vLLM/vLLM-Omni rollout server via `LLMServerClient`, optionally computes async rewards via Ray reward-loop actors, and reassembles everything into a training `DataProto` (or TransferQueue rows in TQ mode).

## Design
- **Batch worker / per-sample loop split** (mirrors veRL's `AgentLoopManager` pattern):
  - `DiffusionAgentLoopWorker` (`diffusion_agent_loop.py`): batch orchestrator. Builds `sampling_params` from `DiffusionRolloutConfig` (`pipeline`, `algo`, `val_kwargs`), derives per-row seeds, spawns `asyncio` tasks via `_run_agent_loop`, post-processes with `_agent_loop_postprocess`/`_postprocess`.
  - `DiffusionAgentLoopOutput` (pydantic `BaseModel`) / `_InternalDiffusionAgentLoopOutput`: per-sample output schema (padded `prompt_ids`, `response_diffusion_output` as NCHW/NTCHW uint8 pixels or float latents, `response_logprobs`, `reward_score`, `metrics`, `extra_fields`).
  - `CompositeAgentLoopWorker` (`composite_agent_loop.py`): subclass for two-stage LLM-CoT→DiT rollouts. Splits reward handles into DiT (`reward_loop_worker_handles[0]`) and AR (`[1:]`) pools; `CompositeAgentLoopOutput` adds `llm_response_logprobs` and `llm_reward_score`; `_postprocess` emits `rollout_llm_log_probs` and `llm_rm_scores` into the batch.
  - `DiffusionAgentLoopWorkerTQ` (`diffusion_agent_loop_tq.py`, `@ray.remote`): TransferQueue-backed variant; `generate_sequences` is fire-and-forget, each session is written via `tq.async_kv_batch_put` with status tags; `create_diffusion_agent_loop_manager` builds veRL's `AgentLoopManagerTQ` with this worker class (wrapped with `@auto_await`).
- **Registered per-sample agent loops** (veRL `AgentLoopBase` + `@register`): `DiffusionSingleTurnAgentLoop` (`single_turn_agent_loop.py`, registered as `"diffusion_single_turn_agent"`) handles prompt rendering, multi-modal extraction, continuous-token initial prompt building (`ct_build_initial_tokens`), per-encoder tokenization (`_tokenize_per_encoder` via `extra_tokenizer_map`), and prompt-embed-cache affinity routing (`_get_routing_request_id`). `MiniMaxH3DiffusionSingleTurnAgentLoop` is imported from `pipelines/minimax_h3_diffusion_nft/agent_loop` in `__init__.py` to fire its registration (avoiding an import cycle).
- **Seed determinism** (`utils.py`): `_derive_rollout_seed` maps `(base_seed, index)` → 63-bit vLLM seed; `maybe_per_rollout_seeds` prefers trainer-provided `_rollout_seed_global_idx` so multi-worker splits keep RNG state stable.
- `_pad_prompt_extra_field` pads `prompt_embeds`/masks/`prompt_token_tags` tensors to `max_prompt_embed_length` (errors if over).

## Flow
1. Trainer calls `generate_sequences(batch)`. Validation batches merge `val_kwargs` sampling params and use a fixed `val_kwargs.seed`; training batches set `global_steps` and derive per-row seeds.
2. Each row is spawned as an asyncio task; `_run_agent_loop` instantiates the registered agent loop (hydra `instantiate` from `_agent_loop_registry`) and calls `loop.run(sampling_params, **row_kwargs)`.
3. `DiffusionSingleTurnAgentLoop.run` extracts images/videos/audios, builds initial prompt (+ negative prompt) ids, optionally tokenizes per extra text encoder, then calls `server_manager.generate(...)` under `simple_timer("generate_sequences")`.
4. `_agent_loop_postprocess` pads prompt ids and extra tensors, then `_compute_score` (if reward handles exist) builds a single-sample `DataProto` and calls `reward_loop_worker_handle.compute_score.remote(...)` (random worker selection; composite variant computes both DiT and AR scores).
5. `_postprocess` concatenates per-sample outputs into `DataProto(batch, non_tensor_batch, meta_info)` with `prompts`, `responses`, optional `rollout_log_probs`, `rm_scores` (+`llm_rm_scores`/`rollout_llm_log_probs` for composite), `__num_turns__`, reward extra keys, and per-sample `metrics`.
6. TQ mode instead streams each finished session into TransferQueue partitions `train`/`val` with uid-keyed status tags.

## Integration
- Consumed by: `trainer/` (Ray trainer constructs workers via veRL's `AgentLoopManager`/`AgentLoopManagerTQ`), pipeline-specific loops registered into `_agent_loop_registry`.
- Depends on: `verl.experimental.agent_loop` (`AgentLoopBase`, `register`, `_agent_loop_registry`, `get_trajectory_info`), `verl.protocol.DataProto`, `verl.workers.rollout.llm_server.LLMServerClient`, `verl_omni.workers.config` (`DiffusionModelConfig`, `DiffusionRolloutConfig`), `verl_omni.agent_loop.utils`, `transfer_queue`, `ray`, `hydra`.
- Key entry points: `DiffusionAgentLoopWorker.generate_sequences`, `CompositeAgentLoopWorker.generate_sequences`, `DiffusionSingleTurnAgentLoop.run`, `DiffusionAgentLoopWorkerTQ.generate_sequences`, `create_diffusion_agent_loop_manager`.
