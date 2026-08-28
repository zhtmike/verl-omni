# verl_omni/workers/rollout/

## Responsibility
Rollout-side plumbing for verl-omni: registers the `vllm_omni` rollout backend in verl's rollout registry, defines the `DiffusionOutput` result contract, registers `vLLMOmniReplica` in the `RolloutReplicaRegistry`, and provides the diffusion-aware retry client used by agent loops. The concrete async server lives in `vllm_rollout/` (see sub-map).

## Design
- `base.py`: one-line extension of verl's `_ROLLOUT_REGISTRY` — maps `("vllm_omni", "async")` → `verl.workers.rollout.vllm_rollout.ServerAdapter` (verl's adapter resolves HTTP-server replicas; the replica class itself is overridden via `RolloutReplicaRegistry` here). This is what makes `rollout.name=vllm_omni, rollout.mode=async` resolve in `get_rollout_class`.
- `replica.py`:
  - `DiffusionOutput(BaseModel)` (pydantic, `arbitrary_types_allowed`) — the diffusion generate result: `diffusion_output` (uint8 pixel tensor CHW/TCHW in [0,255], or float latents), `log_probs`, `stop_reason` (`completed`/`aborted`/None), `num_preempted`, `extra_fields` (dynamic: `all_latents`, `all_timesteps`, `global_steps`, prompt embeddings, audio, ...).
  - Registers `"vllm_omni"` in verl's `RolloutReplicaRegistry` with lazy loader `_load_vllm_omni` → `vLLMOmniReplica`.
- `diffusion_llm_server.py`: `DiffusionWholeSampleRetryLLMServerClient(LLMServerClient)` — wraps verl's async server client; on `stop_reason ∈ ("aborted", "abort")` retries the whole sample (original prompt/sampling params) up to `max_retries` (default 50) with `retry_wait_s` backoff; tracks `min_global_steps`/`max_global_steps` across attempts into `extra_fields` so off-policy staleness metrics stay correct.

## Flow
1. `ActorRolloutRefWorker.init_model` calls `get_rollout_class("vllm_omni", "async")` → verl's `ServerAdapter` (registered by `base.py`).
2. The adapter consults `RolloutReplicaRegistry` → `vLLMOmniReplica` (`rollout/vllm_rollout/vllm_omni_async_server.py`), which spawns `vLLMOmniHttpServer` ray actors per replica.
3. Generation requests flow through `LLMServerClient.generate` (or the `DiffusionWholeSampleRetryLLMServerClient` wrapper) → HTTP → `vLLMOmniHttpServer.generate` → `DiffusionOutput`/`TokenOutput`; aborted diffusion samples are retried whole-sample client-side.
4. Weight sync reaches the replica via `rollout.update_weights` / `_execute_method("update_weights_from_ipc")` from `ActorRolloutRefWorker.update_weights`, or via `OmniCheckpointEngineManager` in disaggregated mode.

## Integration
- Consumed by: `verl_omni/workers/engine_workers.py` (`get_rollout_class`, `self.rollout`), verl's `ServerAdapter`, agent-loop server clients.
- Depends on: `verl.workers.rollout.base` (`_ROLLOUT_REGISTRY`), `verl.workers.rollout.replica` (`RolloutReplicaRegistry`, `RolloutMode`, `TokenOutput`), `verl.workers.rollout.llm_server.LLMServerClient`, `verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server`.
- Sub-maps: `vllm_rollout/codemap.md`.
- Key entry points: `DiffusionOutput`, `DiffusionWholeSampleRetryLLMServerClient`, the `("vllm_omni", "async")` registry entry.
