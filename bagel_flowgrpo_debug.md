# BAGEL FlowGRPO Parity Debug Ledger

## Scope

Compare the current `verl-omni` BAGEL LoRA FlowGRPO path with the official
`flow_grpo` `pickscore_bagel_lora` path. Each probe records the command,
environment, runtime target, and result.

## Environment

- Workspace: local `verl-omni` checkout.
- Official FlowGRPO: sibling/local `flow_grpo` checkout.
- Reference verl: sibling/local `verl` checkout.
- vLLM-Omni: sibling/local `vllm-omni` checkout.
- GPU session: tmux session `mike`
- verl-omni env: `verl-omni-022`
- official env: `flow_grpo`

## Latest Run Summary

- Log: `train_bagel_flowgrpo.log`
- Branch: `bagel_debug`
- Latest parity commit: `0206ef6 fix: align BAGEL FlowGRPO parity`
- Latest rerun observed: step 1 completed after relaunching with
  `sde_window_seed: None`.
- Per-step runtime: about 472 seconds for the first completed update.
- Primary implication: the current real training run looks healthy after the
  SDE-window default fix; strict parity probes should still use tiny batches
  and fixed seeds.

## Static Parity Matrix

| Area | Official `flow_grpo` | `verl-omni` port | Status |
| --- | --- | --- | --- |
| Entry point | `accelerate launch ... scripts/train_bagel.py --config config/grpo.py:pickscore_bagel_lora` | `python3 -m verl_omni.trainer.main_diffusion` via `examples/flowgrpo_trainer/run_bagel_flowgrpo_lora.sh` | Different frameworks, same intended task. |
| GPUs | `gpu_number = 8` | `trainer.n_gpus_per_node=4` | Scaled local run. |
| Model | `ByteDance-Seed/BAGEL-7B-MoT` | same local path | Match. |
| Resolution | `512` | `height=512`, `width=512` from resolved config | Match. |
| Samples per prompt | `num_image_per_prompt=16` | `actor_rollout_ref.rollout.n=16` | Match. |
| Train batch | per-GPU sample batch `6`, effective prompt batch derived from `num_batches_per_epoch` | `data.train_batch_size=48`, mini/micro `24` | Similar effective scale, implementation differs. |
| Timesteps | `num_steps=15`, `eval_num_steps=50`; BAGEL code uses `torch.linspace(1, 0, num_timesteps)` then drops terminal zero | `pipeline.num_inference_steps=15`, val `50`; `setup_bagel_sigmas()` and `vllm_omni_num_timesteps()` compensate vLLM-Omni step convention | Needs direct schedule probe. |
| Timestep shift | inherited `train.timestep_shift=3.0` | `BAGEL_TIMESTEP_SHIFT=3.0` | Match by config. |
| CFG | `cfg_text_scale=4.0`, `cfg_img_scale=1.0`, `cfg_interval=[0,1]`, global renorm | `BAGEL_FLOWGRPO_CFG_DEFAULTS` forces same defaults for rollout/training | Match by code, needs tensor parity. |
| SDE noise | `noise_level=1.3`, `sde_window_size=2`, `sde_window_range=(0, num_steps//2)` | `noise_level=1.3`, `sde_window_size=2`, `sde_window_range=[0,7]` | Config match; boundary sampling needs probe. |
| SDE window sampling | `random.seed(process_index)` then `randint(low, high - size)` | `_pick_sde_window(..., seed=req.sampling_params.seed, request_id=...)` then `randint(low, high - size)` | Range formula matches; seed source differs. |
| Rollout forward | in-process `InterleaveInferencer` and `Bagel.generate_image()` | vLLM-Omni `BagelPipelineWithLogProb` with scheduler adapter | Major architecture difference. |
| Rollout trajectory | returns `all_latents`, `all_log_probs`, `all_timesteps` from SDE window | returns `all_latents`, `all_log_probs`, `all_timesteps` in `custom_output` | Needs shape/order parity probe. |
| Training forward | official `generate_image_learn()` recomputes log-prob inside same BAGEL implementation | FSDP `BagelForTraining` via `BagelDiffusion.forward_and_sample_previous_step()` | Major architecture difference. |
| Attention in denoise | official `_forward_flow(... is_causal=False)` for denoising | latest commit changed training attention to unified `is_causal=False` with padding mask | Likely aligned, needs log-prob/tensor probe. |
| Text generation attention | official AR text path uses `is_causal=True` | not used for PickScore t2i training-side denoise | Not a target for this parity run. |
| LoRA config | `r=64`, `alpha=128`, Gaussian init; targets `self_attn.{q,k,v,o}_proj_moe_gen` and `mlp_moe_gen.*` on `model.language_model` | `r=64`, `alpha=128`, Gaussian init, `lora_dtype=float32`; targets omit `self_attn.` because `BagelForTraining` has flattened module names under `layers.*` | Needs name/count mapping probe. |
| LoRA dtype | official casts LoRA params to `inference_dtype` (`bf16`) after PEFT init | local config sets `lora_dtype=float32` | Potential difference; verify actual trainable dtype. |
| Reward | `PickScoreScorer`, `scores.diag() / 26` | `pickscore_reward.compute_score`, same CLIP/PickScore models and `/ 26` normalization | Match by code. |
| Advantage | `PerPromptStatTracker(global_std=False)` uses growing per-prompt history | `compute_flow_grpo_outcome_advantage(global_std=False)` normalizes within current batch group | Potential behavioral difference if official history spans steps. |
| PPO loss | `ratio = exp(log_prob - sample["log_probs"][i])`, clip `1e-5`, beta `0` | `FlowGRPOLoss`, same ratio/clip form, no KL loss | Formula match. |
| Rollout correction | none in official single-process setup | enabled: recompute `old_log_probs`, sequence IS metrics, PPO loss still uses actor ratio | Intentional architectural addition; current metrics show rollout/training log-probs are close. |

## Probe Results

### Probe 1: CPU Schedule, Window, And Config Checks

Command environment:

```bash
conda activate verl-omni-022
PYTHONPATH="<verl-omni>:<vllm-omni>:$PYTHONPATH" python ...
```

Results:

- Official BAGEL schedule length for `num_steps=15`: `14`.
- `verl-omni` `setup_bagel_sigmas()` length for `num_steps=15`: `14`.
- `vllm_omni_num_timesteps(15)`: `14`.
- Max absolute schedule difference: `0.0`.
- First/last official timestep: `1.0`, `0.1875`.
- First/last local timestep: `1.0`, `0.1875`.
- Official first/last `dt`: `-0.02500009536743164`, `-0.1875`.
- SDE window bounds with `range=(0,7)`, `size=2` match official for seeds `0..7`.
- Request-id based windows are deterministic but not process-index aligned:
  - request id `0`: `(2, 4)`
  - request id `1`: `(3, 5)`
  - request id `prompt-0-sample-0`: `(2, 4)`

Official config dump from `flow_grpo`:

- model: `ByteDance-Seed/BAGEL-7B-MoT`
- resolution: `512`
- steps/eval steps: `15` / `50`
- guidance/eval guidance: `4.0` / `4.0`
- use LoRA: `True`
- sample train batch size: `6`
- images per prompt: `16`
- batches per epoch: `16`
- train batch size: `6`
- gradient accumulation: `8`
- clip: `1e-05` / `1e-05`
- beta / learning rate: `0.0` / `0.0001`
- mixed precision: `bf16`
- noise/window: `1.3`, `2`, `(0, 7)`
- global std: `False`
- reward: `{"pickscore": 1.0}`
- per-prompt stat tracking: `True`
- activation checkpointing / optimizer offload: `True` / `True`

Conclusion: schedule and seed-based SDE window bounds are aligned. Configs match
the intended hyperparameters. Remaining CPU-visible concern is advantage
normalization scope: official `PerPromptStatTracker` stores per-prompt history,
while `verl-omni` computes group stats on the current rollout batch.

### Probe 2: GPU Prompt And LoRA Checks

Command environment:

```bash
tmux session mike
conda activate verl-omni-022
python scripts/bagel_prompt_lora_probe.py > bagel_gpu_prompt_lora_probe.log
```

Results from `bagel_gpu_prompt_lora_probe.log`:

- Prompt sample: `a photograph of patterdale dog driving a land rover car toy in the cave lava,terrier`
- Stored parquet prompt length: `21` tokens.
- Stored first/last token IDs: `151644`, `151645`.
- Local parquet helper match: `True`.
- vLLM `prepare_prompts`-style match: `True`.
- Official `prepare_prompts`-style match: `True`.
- Decode -> extract -> retokenize match: `True`.
- LoRA rank/alpha target: `64` / `128`.
- vLLM packed mapping confirms `q/k/v *_moe_gen` are packed under `qkv_proj_moe_gen`.
- vLLM packed mapping confirms `mlp_moe_gen.gate_proj` and `mlp_moe_gen.up_proj` are packed under `mlp_moe_gen.gate_up_proj`.
- Confirmed dtype difference: local run config sets `actor_rollout_ref.model.lora_dtype=float32`; official casts LoRA params to bf16 after PEFT injection.

Conclusion: prompt tokenization is not the current parity issue. LoRA module
targets are structurally aligned, but LoRA dtype remains a deliberate or
accidental difference to evaluate later.

### Probe 3: GPU Rollout And Log-Prob Checks

Command environment:

```bash
tmux session mike, original pane with CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/bagel_mini_train_probe.sh > bagel_gpu_mini_train_probe_batch4.log
bash scripts/bagel_mini_train_probe.sh > bagel_gpu_mini_train_probe_prob3_clean.log
```

Probe notes:

- A new tmux window did not inherit the Slurm CUDA environment and failed with
  `Total available GPUs 0 is less than total desired GPUs 4`; rerunning in the
  original `mike` pane fixed GPU visibility.
- A 1-sample mini batch reached rollout/reward but failed in old-log-prob
  dispatch because batch length `1` is not divisible by data-parallel size `4`.
- The adjusted 4-sample mini batch reached:
  - vLLM-Omni server startup,
  - agent loop rollout,
  - PickScore reward model load,
  - old-log-prob / actor loss path (`FlowGRPOLoss` warning at ratio std with
    scalar per-rank microbatches),
  - checkpoint creation at `global_step_1`.
- The run did not exit cleanly inside the intended timeout because the trainer
  still runs validation at the final step whenever `trainer.test_freq > 0`;
  `trainer.test_freq=999` therefore did not disable final validation. It was
  interrupted manually and `ray stop --force` was run in tmux.
- `scripts/bagel_mini_train_probe.sh` was updated to use `trainer.test_freq=0`
  and `trainer.save_freq=0`, so future mini probes stop after the train/update
  boundary instead of entering final validation/checkpointing.
- A first rerun completed with `MINI_PROBE_EXIT=0`, but resumed from the prior
  `global_step_1` checkpoint. The probe wrapper was then updated with
  `trainer.resume_mode=disable` to force clean one-step runs.
- Clean rerun log: `bagel_gpu_mini_train_probe_prob3_clean.log`.
  - Resolved config confirms `resume_mode='disable'`, `test_freq=0`, and
    `save_freq=0`.
  - No `Found checkpoint` / `Resuming from` lines were emitted.
  - Final console metrics were captured at `step:1`, including
    `rollout_corr/training_ppl=4.728292942047119`,
    `rollout_corr/training_log_ppl=1.553560733795166`, and
    `rollout_corr/kl=0.0019817352294921875`.
  - The trainer printed `Final validation metrics: None`.
  - The wrapper appended `MINI_PROBE_EXIT=0`.

Conclusion: the local rollout -> reward -> old-log-prob -> actor-loss path is
exercisable with the 4-sample mini probe and now completes cleanly from scratch.
The earlier hang was final validation/checkpoint behavior, not a rollout or
log-prob deadlock.

### Parity Audit: Rollout, Log-Prob, And Actor Update

Static code paths checked:

- Official rollout: `flow_grpo/scripts/train_bagel.py` calls
  `InterleaveInferencer`, then `Bagel.generate_image()` records
  `all_latents`, `all_log_probs`, and `all_timesteps`.
- Official log-prob / actor update: `Bagel.generate_image_learn()` recomputes
  `log_prob` for each stored `latents[i] -> prev_latents[i]` transition and
  applies clipped FlowGRPO loss.
- `verl-omni` rollout: `BagelPipelineWithLogProb.forward()` records vLLM-Omni
  trajectory tensors and slices `latents[begin:end+1]`,
  `timesteps[begin:end]`, and in-window rollout log-probs.
- `verl-omni` log-prob / actor update:
  `BagelDiffusion.forward_and_sample_previous_step()` scores
  `all_latents[:, step] -> all_latents[:, step + 1]`, and
  `FlowGRPOLoss` applies the clipped objective.

Matches:

- Trajectory order matches: both implementations train on the latent before
  SDE step `i`, the latent after step `i`, and the timestep `t_i`.
- SDE transition mean/std match the official BAGEL formula:
  `std=sqrt(t / (1 - where(t == 1, sigma_max, t))) * noise_level`, with
  `prev_sample_mean = sample*(1 + std^2/(2t)*dt) + v_t*(1 + std^2*(1-t)/(2t))*dt`.
- Actor loss matches in form:
  `ratio=exp(new_log_prob - old_log_prob)`,
  `max(-adv*ratio, -adv*clamp(ratio, 1-clip, 1+clip))`, mean reduction,
  and `clip=1e-5`.
- Advantage grouping is closer than first suspected: official
  `PerPromptStatTracker` is cleared after each gathered rollout batch, so it is
  not persistent across epochs. `verl-omni` groups repeated samples by the
  original batch `uid` after `rollout.n` expansion. This is equivalent for the
  intended repeated-prompt sampler, except duplicate prompt text from distinct
  dataset rows would be grouped by official but not by `uid`.

Differences / risks:

- Log-prob convention differs by a model-independent Gaussian normalization
  constant. Official BAGEL records only the quadratic term and averages it;
  `verl-omni`'s scheduler includes `-log(std * sqrt(-dt)) - log(sqrt(2*pi))`
  before averaging. This cancels in `new - old` ratios when both sides use the
  same convention, but raw official-vs-`verl-omni` log-probs and rollout
  correction diagnostics are not directly comparable without accounting for the
  constant.
- The normal `verl-omni` BAGEL script enables decoupled rollout correction
  (`algorithm.rollout_correction.rollout_is=sequence`). Official BAGEL has no
  separate vLLM rollout engine and no IS multiplier in the actor loss. With the
  current local run, rollout correction weights are close to 1 but still change
  the objective slightly.
- LoRA dtype remains different: official casts LoRA params to bf16, while the
  local script sets `actor_rollout_ref.model.lora_dtype=float32`.

### Probe 4: Nonzero Actor Update Check

Command environment:

```bash
tmux session mike, original pane with CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/bagel_mini_train_probe.sh actor_rollout_ref.rollout.n=2 > bagel_gpu_mini_train_probe_n2_minib4.log
```

Probe notes:

- `scripts/bagel_mini_train_probe.sh` now accepts trailing Hydra overrides via
  `"$@"`, so small parity probes can change `rollout.n` without duplicating the
  wrapper.
- A first `n=2` attempt with `actor_rollout_ref.actor.ppo_mini_batch_size=8`
  failed at actor update because each data-parallel rank received local batch
  size `2`, while the derived local mini-batch size was `4`.
- The corrected `n=2` run kept the wrapper's `ppo_mini_batch_size=4` and
  completed with `MINI_PROBE_EXIT=0`.
- Final metrics from `bagel_gpu_mini_train_probe_n2_minib4.log`:
  - `critic/rewards/group_size=2.0`
  - `critic/rewards/zero_std_ratio=0.0`
  - `critic/advantages/max=0.8639509081840515`
  - `critic/advantages/min=-0.8639509081840515`
  - `actor/loss=0.0001386702060699463`
  - `actor/grad_norm=0.006026435177773237`
  - `rollout_corr/log_ppl_diff=0.0015356987714767456`
  - `rollout_corr/rollout_is_mean=0.9969344735145569`

Conclusion: the `n=2` probe exercises a real nonzero actor update. The path is
functional, but it also confirms that rollout correction is active and slightly
modifies the objective relative to official BAGEL.

### Probe 5: Strict One-Step Parity Attempt

Goal: run one controlled update on both sides with the closest available
settings, then compare rollout/log-prob/loss/gradient metrics.

Wrappers added:

```bash
scripts/bagel_strict_parity_probe.sh
scripts/bagel_official_one_step_probe.sh
```

`verl-omni` strict settings:

- `actor_rollout_ref.rollout.n=2`
- `algorithm.rollout_correction.bypass_mode=True`
- `algorithm.rollout_correction.rollout_is=null`
- `algorithm.rollout_correction.rollout_rs=null`
- `actor_rollout_ref.model.lora_dtype=bfloat16`
- `actor_rollout_ref.rollout.seed=0`
- `actor_rollout_ref.rollout.algo.sde_window_seed=0`

`verl-omni` strict result:

- Log: `bagel_gpu_strict_parity_verl.log`
- Exit marker: `STRICT_VERL_EXIT=0`
- Config confirms bypass mode, no rollout IS, bf16 LoRA, `n=2`, and SDE seed
  `0`.
- Final metrics:
  - `actor/ppo_kl=0.001769721508026123`
  - `actor/pg_clipfrac=1.0`
  - `actor/ratio_mean=0.9982332475483418`
  - `actor/loss=0.0007517524063587189`
  - `actor/grad_norm=0.0078125`
  - `critic/rewards/group_size=2.0`
  - `critic/rewards/zero_std_ratio=0.0`
  - `critic/advantages/max=0.8640219569206238`
  - `critic/advantages/min=-0.8640233874320984`

Official one-step result after environment repair:

- Command wrapper: `scripts/bagel_official_one_step_probe.sh`
- Log: `bagel_gpu_strict_parity_official.log`
- Exit marker: `OFFICIAL_PROBE_EXIT=0`
- Environment fixes required:
  - isolate Python user site with `PYTHONNOUSERSITE=1`;
  - preload env-local NCCL:
    `${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2`,
    avoiding the older `~/.local/.../nvidia/nccl/lib/libnccl.so.2` that caused
    `ncclCommShrink` import failure;
  - install matching FlashAttention wheel
    `flash-attn==2.8.3+cu128torch2.10`;
  - add compatibility patches for the official repo's current dependency set:
    explicit Qwen2 `pad_token_id=None`, a local `"default"` RoPE initializer,
    and PickScore feature extraction for `BaseModelOutputWithPooling`.
- Runtime memory fix: the default official FSDP config OOMed on 4 visible H800s
  during backward unshard. The probe now uses
  `scripts/accelerate_fsdp_cpu_offload.yaml` (`fsdp_offload_params: true`,
  no forward/backward prefetch) and completes within the 10 minute timeout.
- Final official metrics:
  - `group_size=2`
  - `clipfrac=0`
  - `loss=0`
  - `policy_loss=0`
  - `kl_loss=-1` (`beta=0`, no reference KL path)
  - `reward_avg=0.78049`
  - `reward_pickscore=0.78049`
  - `reward_std_mean=0.03131`
  - `trained_prompt_num=2`

Conclusion: both sides now run a one-step `n=2` e2e. The official first update
logs zero scalar policy/loss value, consistent with recomputing log-probs under
the same model before any optimizer update. The strict `verl-omni` run still
shows a small old/new rollout-logprob mismatch (`actor/ppo_kl=0.0017697`,
`actor/ratio_mean=0.998233`) even with rollout correction bypassed, so the next
parity target is the rollout/log-prob boundary rather than reward or advantage
normalization.

## Findings

- Confirmed match: BAGEL timestep schedule and `vllm_omni_num_timesteps(15)` compensate vLLM-Omni to the same 14 denoise timesteps used by official BAGEL.
- Confirmed match: prompt IDs in parquet match official/vLLM BAGEL tokenization and survive decode/retokenize.
- Confirmed match: PickScore implementation and `/26` normalization match official code.
- Confirmed likely match: latest attention commit aligns denoise attention with official `is_causal=False`; the mini probe reached loss without attention shape errors.
- Confirmed clean GPU path: the 4-sample Probe 3 rerun reached rollout, reward, old-log-prob, actor update, final metrics, and exit code `0` with final validation/checkpointing disabled.
- Confirmed nonzero actor update: the `n=2` Probe 4 run produced nonzero advantages, actor loss, and grad norm.
- Confirmed gap: BAGEL rollout adapter ignored `sample_strategy`, `iters_per_group`, and `sde_window_seed` even though those knobs exist in `DiffusionRolloutAlgoConfig`; the Qwen MixGRPO adapter already honors them.
- Fixed gap: BAGEL raw log-prob values differed from official by the Gaussian normalization constant; BAGEL now opts into official's quadratic-only log-prob convention while the generic scheduler default remains unchanged.
- Remaining gap: rollout correction is enabled in the `verl-omni` BAGEL script, but official BAGEL has no IS multiplier in the actor objective.
- Clarified: official advantage normalization is per gathered rollout batch for this script because `stat_tracker.clear()` is called after each update; `verl-omni`'s `uid` grouping is equivalent for repeated samples from the same dataset row.
- Remaining gap: official LoRA params are cast to bf16, while the current local BAGEL run config uses float32 LoRA.
- Confirmed official e2e path after environment repair: rollout, PickScore
  reward, log-prob recompute, backward, optimizer step, and W&B summary complete
  with `OFFICIAL_PROBE_EXIT=0`.
- Remaining parity signal: official first-update scalar `policy_loss/loss=0`,
  while strict `verl-omni` reports small but nonzero old/new log-prob drift
  (`actor/ppo_kl=0.0017697`).

## Next Actions

- Applied scoped fix in `verl_omni/pipelines/bagel_flow_grpo/vllm_omni_rollout_adapter.py`: BAGEL now honors deterministic random and progressive SDE-window strategy controls.
- Verified the SDE-window strategy helper on CPU:
  - seeded random step 0: `(3, 5)`
  - seeded random step 1: `(1, 3)`
  - request-id fallback: `(0, 2)`
  - progressive step 0: `(0, 2)`
  - progressive step 2: `(4, 6)`
  - progressive clamp: `(5, 7)`
- No linter errors were reported for the edited adapter and probe files.
- Recommended next validation: run a short strict-parity config with rollout
  correction disabled and compare reward trend / `actor/pg_clipfrac`, then
  decide whether to align LoRA dtype to bf16 or keep float32 for stability.
- Next parity check: instrument one identical prompt/sample to dump rollout
  timesteps, SDE window, old log-probs, recomputed log-probs, and first LoRA
  gradient norm on both sides. The official env is no longer the blocker.

## 2026-06-24 Parity Dump and Log-Prob Normalizer Fix

Added probe-gated JSONL dumps controlled by `BAGEL_PARITY_DUMP_DIR`:

- Official `flow_grpo`: `flow_grpo/bagel/modeling/bagel/bagel.py` dumps rollout, learn-step old/new log-probs, ratios, and post-backward grad norms; `flow_grpo/scripts/train_bagel.py` dumps prompt/reward/advantage context.
- `verl-omni`: BAGEL rollout dumps are written from `verl_omni/pipelines/bagel_flow_grpo/vllm_omni_rollout_adapter.py`; actor recompute dumps are written from `verl_omni/workers/engine/fsdp/diffusers_impl.py`.
- Probe wrappers now create default dump directories under `bagel_parity_dumps/official` and `bagel_parity_dumps/verl`.

First dump comparison found a concrete log-prob convention mismatch:

- Official BAGEL `_sde_step_with_logprob` uses only the quadratic term:
  `-((prev_sample - prev_sample_mean) ** 2) / (2 * std**2)`, then averages.
- `verl-omni`'s shared `FlowMatchSDEDiscreteScheduler` included the Gaussian
  normalizer terms (`-log(std)` and `-log(sqrt(2*pi))`).
- This caused matching `[1, 3]` windows / timesteps to report different raw
  rollout log-prob scales: official around `-0.50`, `verl-omni` around
  `-1.72/-1.38`.

Applied scoped fix:

- `FlowMatchSDEDiscreteScheduler.step/sample_previous_step` now accept
  `include_logprob_normalizer=True` by default, preserving generic scheduler
  behavior.
- BAGEL rollout and training adapters pass `include_logprob_normalizer=False`
  to match official BAGEL's quadratic-only convention.

Verification:

- Official instrumented probe completed with `OFFICIAL_PROBE_EXIT=0`.
  Representative official window `[1, 3]`: timesteps
  `[0.9749999, 0.9473684]`, rollout log-probs
  `[-0.5002986, -0.5018647]`, recompute ratio exactly `1.0`.
- `verl-omni` strict probe after the fix completed with `STRICT_VERL_EXIT=0`.
  Representative matching window `[1, 3]`: timesteps
  `[0.9749999, 0.9473684]`, rollout log-probs now around
  `[-0.4999826, -0.4978249]` / `[-0.4972601, -0.5005839]`, i.e. on the
  official scale.
- Remaining parity gap is no longer the raw log-prob definition. It is the
  small old/new recompute drift inside `verl-omni`: latest strict run reports
  `actor/ppo_kl=0.00176969`, `actor/ratio_mean=0.99823329`,
  `actor/loss=0.00075174`, and `actor/grad_norm=0.00726318`.

Next target: inspect why `verl-omni` recomputed training log-probs differ from
its own rollout log-probs after the convention fix. The likely area is
rollout-vs-training forward parity (CFG branch, attention/caching, latent
packing, or train/eval flags), not reward/advantage or log-prob reduction.

## 2026-06-24 Velocity Dump and Cache-Semantics Fix

Added focused `verl-omni` BAGEL dumps:

- Rollout adapter now emits `rollout_step` records with `sample`,
  `model_output` (velocity), `prev_sample`, `prev_sample_mean`, `std_dev_t`,
  and `log_prob`.
- Training adapter now emits `train_detail` records with raw conditional
  velocity, text-unconditional velocity, final CFG velocity, latent inputs,
  `prev_sample_mean`, and scheduler scalars.
- FSDP actor train-step dumps now also include `prev_sample_mean`, `std_dev_t`,
  and `sqrt_dt`.

Finding:

- Same-trajectory rollout/training records had matching `sample`,
  `prev_sample`, timestep, `std_dev_t`, and scheduler scalars, but the
  recomputed training velocity differed before the scheduler.
- The main structural mismatch was the prompt/text path: vLLM-Omni BAGEL
  rollout precomputes prompt text into a causal KV cache, then denoises over
  image markers + latent tokens. The training adapter previously ran prompt
  text + image tokens in one fully bidirectional denoise sequence, allowing
  prompt token states to absorb latent information across layers.

Applied scoped fix:

- `verl_omni/pipelines/bagel_flow_grpo/bagel_model.py` now uses an asymmetric
  attention mask in BAGEL training forward:
  prompt-token queries are causal/text-only, while image-marker and latent
  queries still attend bidirectionally to prompt + image tokens. This better
  matches rollout's cached-prompt semantics without changing the rollout path.

Verification:

- Strict `verl-omni` probe completed with `STRICT_VERL_EXIT=0`.
- Old/new drift improved substantially:
  - before cache-semantics fix: `actor/ppo_kl=0.00176969`,
    `actor/ratio_mean=0.99823329`, `actor/loss=0.00075174`,
    `actor/grad_norm=0.00765991`;
  - after fix: `actor/ppo_kl=9.31956e-05`,
    `actor/ratio_mean=0.99990684`, `actor/loss=5.07049e-05`,
    `actor/grad_norm=0.00451660`.
- Representative post-fix per-step old/new deltas are now small:
  `-0.5004330 -> -0.5004470`, `-0.4996610 -> -0.4997612`,
  `-0.4999826 -> -0.5000122`, `-0.4978249 -> -0.4979298`.

Residual gap:

- The remaining `~1e-4` KL is much smaller but not exactly zero. Remaining
  suspects are low-level numeric differences between custom actor forward and
  vLLM-Omni rollout forward, including packed-cache implementation details,
  LoRA dtype/weight loading path, and attention kernel precision.

## 2026-06-24 Final Parity Analysis Stop Point

Final strict `verl-omni` probe already includes the intended config-only
alignment knobs:

- `actor_rollout_ref.model.lora_dtype=bfloat16`, matching official BAGEL LoRA
  dtype.
- `algorithm.rollout_correction.bypass_mode=True`, so PPO old log-probs are
  the rollout log-probs, matching official's in-process rollout/update setup.
- `algorithm.rollout_correction.rollout_is=null` and
  `algorithm.rollout_correction.rollout_rs=null`, so no extra IS/RS weights are
  applied to the actor objective.
- Fixed rollout seed and SDE-window seed for reproducibility.

Final comparison:

- Official one-step probe completes with `OFFICIAL_PROBE_EXIT=0`. W&B summary
  reports `policy_loss=0`, `loss=0`, and `kl_loss=-1` for the first update,
  consistent with exact old/new log-prob equality in the official in-process
  implementation.
- Strict `verl-omni` final probe completes with `STRICT_VERL_EXIT=0`.
  Key metrics after all parity fixes:
  `actor/ppo_kl=9.31956e-05`, `actor/ratio_mean=0.99990684`,
  `actor/loss=5.07049e-05`, `actor/grad_norm=0.00451660`.
- The final strict probe uses rollout log-probs as the old-policy anchor, so
  the remaining drift is specifically current actor recompute vs rollout actor
  forward, not old-policy recompute, reward, advantage normalization, or the
  SDE log-prob normalizer.

Conclusion:

- The meaningful training-process inconsistencies found in this parity pass
  were fixed:
  1. BAGEL rollout/training log-prob convention now matches official
     quadratic-only SDE log-prob.
  2. BAGEL training attention now better matches rollout's cached-prompt
     semantics.
  3. Strict probe config uses bf16 LoRA and disables extra rollout IS/RS
     objective weighting.
- The remaining `~1e-4` KL is small enough to treat as residual numeric /
  implementation drift between the custom FSDP BAGEL actor forward and
  vLLM-Omni's packed cached rollout forward.
- Stop further parity patching here unless a longer training run shows a
  practical instability. The next useful validation is a short real training
  smoke run with the strict-aligned config, not more one-step internals work.

## 2026-06-24 Real Training Rerun Handoff

Context:

- After the parity commit (`0206ef6`), the normal BAGEL rollout config was
  changed so `sde_window_seed` defaults to `null` / `None` instead of `0`.
- This matters for `_pick_strategy_sde_window`: `sde_window_seed=0` made SDE
  window selection deterministic by global step, while `None` restores the
  per-request/random behavior expected for ordinary training runs.
- The user relaunched the real training job in `train_bagel_flowgrpo.log`.
  The resolved config confirms `'sde_window_seed': None`.

Observed first completed update:

- `training/global_step=1`
- `critic/rewards/mean=0.7739937901496887`
- `critic/rewards/max=0.9798278212547302`
- `critic/rewards/min=0.5089152455329895`
- `actor/ppo_kl=7.073622209219366e-05`
- `actor/ratio_mean=0.9999293088912964`
- `actor/grad_norm=0.0002741813659667969`
- `timing_s/step=471.7991578609217`

Interpretation for tomorrow:

- The low initial reward seen before the rerun was most likely caused by the
  old `sde_window_seed=0` default, not by the log-prob normalizer fix or the
  asymmetric attention-mask change.
- Restoring `sde_window_seed=null` brought the first-step reward back to the
  expected high range and kept the actor update small (`ppo_kl ~7e-5`,
  `ratio_mean ~0.99993`).
- This is a good run to let continue. If reward later drops, first inspect the
  W&B reward curve, SDE-window sampling distribution, and generated samples
  before doing more parity patching.
