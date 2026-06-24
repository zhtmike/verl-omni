# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BAGEL (MoT) rollout-side adapter for FlowGRPO.

Extends ``BagelPipeline`` with an SDE scheduler for stochastic denoising
and log-probability recording.  Applies per-request SDE windowing so noise
is only injected on a contiguous subset of denoising steps, matching the
original flow_grpo BAGEL rollout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.bagel_flow_grpo.common import (
    BAGEL_FLOWGRPO_CFG_DEFAULTS,
    maybe_to_cpu,
    setup_bagel_sigmas,
    vllm_omni_num_timesteps,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

logger = logging.getLogger(__name__)


def _parity_tensor_summary(value, limit: int = 8):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return value
    detached = value.detach().float().cpu()
    flat = detached.reshape(-1)
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "values": flat[:limit].tolist(),
    }
    if flat.numel() > 0:
        summary.update(
            {
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()),
                "min": float(flat.min().item()),
                "max": float(flat.max().item()),
            }
        )
    return summary


def _parity_dump(record: dict) -> None:
    dump_dir = os.environ.get("BAGEL_PARITY_DUMP_DIR")
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    payload = {"side": "verl", "rank": rank, "pid": os.getpid(), **record}
    with open(os.path.join(dump_dir, f"verl_rollout_rank{rank}_pid{os.getpid()}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


# TODO: Drop decode→re-tokenize helpers once vllm-omni BagelPipeline accepts
# prompt_token_ids directly (currently only reads text from req.prompts[0]["prompt"]).
_CHAT_MARKERS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def _to_token_list(token_ids: Any) -> list[int] | None:
    if token_ids is None:
        return None
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _extract_prompt_text(decoded: str) -> str:
    if "<|im_start|>" in decoded:
        user_chunks = []
        for segment in decoded.split("<|im_start|>"):
            if not segment.startswith("user"):
                continue
            content = segment[len("user") :].lstrip("\n")
            content = content.split("<|im_end|>", 1)[0]
            user_chunks.append(content)
        if user_chunks:
            decoded = user_chunks[-1]

    for marker in _CHAT_MARKERS:
        decoded = decoded.replace(marker, "")
    return decoded.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()


@dataclass
class _AdapterStepOutput:
    """Adapter output matching what bagel_transformer.generate_image expects."""

    prev_sample: torch.Tensor
    log_prob: torch.Tensor | None


class _BagelSchedulerAdapter:
    """Adapt ``FlowMatchSDEDiscreteScheduler`` to BAGEL's calling convention.

    BAGEL calls ``scheduler.step(v_t, sigma, x_t, dt, **kwargs)`` with 4
    positional args; the diffusers scheduler expects 3.  SDE noise and
    log-prob recording are gated to a per-request window so steps outside
    the window run deterministically (ODE, ``noise_level=0``).
    """

    def __init__(self, inner: FlowMatchSDEDiscreteScheduler):
        self._inner = inner
        self._sde_window: Optional[tuple[int, int]] = None
        self._base_noise_level: float = 0.0
        self._base_return_logprobs: bool = True
        self._step_counter: int = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def begin_forward(
        self,
        sde_window: Optional[tuple[int, int]],
        noise_level: float,
        return_logprobs: bool,
    ) -> None:
        """Reset adapter state before each rollout ``forward`` call.

        Args:
            sde_window: ``(begin, end_exclusive)`` step range where SDE
                noise is injected and log-probs are recorded.  ``None``
                disables windowing (legacy behavior: noise at every step).
            noise_level: SDE noise level to apply inside the window.
            return_logprobs: whether log-probs are requested at all
                (overridden to ``False`` outside the window even when
                ``True`` here).
        """
        self._sde_window = sde_window
        self._base_noise_level = float(noise_level)
        self._base_return_logprobs = bool(return_logprobs)
        self._step_counter = 0

    def step(
        self,
        model_output: torch.Tensor,
        sigma: float | torch.Tensor,
        sample: torch.Tensor,
        dt: float | torch.Tensor,  # noqa: ARG002 — inner derives dt from timestep schedule
        **kwargs,
    ) -> _AdapterStepOutput:
        """Run one denoising step, gating noise and log-probs by the SDE window.

        Args:
            model_output: Velocity prediction ``v_t`` from the model.
            sigma: Current noise level (BAGEL uses raw sigma, not 0-1000).
            sample: Current latent ``x_t``.
            dt: Step size (ignored; derived from the inner scheduler's
                timestep schedule).

        Returns:
            ``(prev_sample, log_prob)`` where ``log_prob`` is a scalar
            (or ``None`` outside the SDE window).
        """
        i = self._step_counter
        in_window = True
        if self._sde_window is not None:
            begin, end = self._sde_window
            in_window = begin <= i < end
            # Outside the SDE window, run deterministic ODE (noise_level=0)
            # and skip log-prob recording (std_dev_t=0 → log(0)=-inf).
            cur_noise_level = self._base_noise_level if in_window else 0.0
            cur_return_logprobs = self._base_return_logprobs and in_window
            kwargs = {
                **kwargs,
                "noise_level": cur_noise_level,
                "return_logprobs": cur_return_logprobs,
            }

        # Rollout latents are (tokens, C); the scheduler expects a batch dim and
        # otherwise averages log-probs over token/channel dims instead of one scalar.
        sample_in = sample.unsqueeze(0)
        model_output_in = model_output.unsqueeze(0)
        if "prev_sample" in kwargs:
            kwargs = {**kwargs, "prev_sample": kwargs["prev_sample"].unsqueeze(0)}

        out = self._inner.step(
            model_output=model_output_in.float(),  # cast bf16→fp32 for scheduler precision
            timestep=sigma,
            sample=sample_in,
            return_dict=False,
            **kwargs,
        )
        self._step_counter += 1
        prev_sample, log_prob, prev_sample_mean, std_dev_t = out[0], out[1], out[2], out[3]
        if in_window and log_prob is not None:
            _parity_dump(
                {
                    "event": "rollout_step",
                    "step": int(i),
                    "sigma": _parity_tensor_summary(sigma),
                    "dt_arg": _parity_tensor_summary(dt),
                    "sample": _parity_tensor_summary(sample, limit=4),
                    "model_output": _parity_tensor_summary(model_output, limit=4),
                    "prev_sample": _parity_tensor_summary(prev_sample.squeeze(0), limit=4),
                    "prev_sample_mean": _parity_tensor_summary(prev_sample_mean.squeeze(0), limit=4),
                    "std_dev_t": _parity_tensor_summary(std_dev_t),
                    "log_prob": _parity_tensor_summary(log_prob),
                }
            )
        prev_sample = prev_sample.squeeze(0)
        if log_prob is not None:
            log_prob = log_prob.reshape(())
        return _AdapterStepOutput(prev_sample=prev_sample, log_prob=log_prob)


def _pick_sde_window(
    window_size: Optional[int],
    window_range: Optional[Any],
    seed: Optional[int],
    request_id: Optional[str],
) -> Optional[tuple[int, int]]:
    """Pick a random contiguous window ``[begin, begin + window_size)``.

    Args:
        window_size: Number of steps in the window.  ``None`` or 0
            disables windowing.
        window_range: ``(low, high)`` inclusive range for the window
            start.  ``None`` defaults to ``[0, window_size)``.
        seed: If set, seed the RNG for reproducibility.
        request_id: If set (and ``seed`` is ``None``), seed the RNG
            with a hash of the request ID so concurrent requests get
            different windows.

    Returns:
        ``(begin, end_exclusive)`` or ``None`` if windowing is disabled.
    """
    if window_size is None or int(window_size) <= 0:
        return None
    if window_range is None:
        return (0, int(window_size))

    low = int(window_range[0])
    high = int(window_range[1])
    high_inclusive = high - int(window_size)
    if high_inclusive < low:
        # Window doesn't fit; clamp to the lowest valid begin.
        return (low, low + int(window_size))

    if seed is not None:
        rng = random.Random(int(seed))
    elif request_id is not None:
        h = hashlib.sha256(str(request_id).encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
    else:
        rng = random.Random()
    begin = rng.randint(low, high_inclusive)
    return (begin, begin + int(window_size))


def _pick_strategy_sde_window(
    *,
    window_size: Optional[int],
    window_range: Optional[Any],
    extra_args: dict[str, Any],
    seed: Optional[int],
    request_id: Optional[str],
) -> Optional[tuple[int, int]]:
    """Pick an SDE window, honoring rollout algo window-strategy knobs.

    BAGEL FlowGRPO historically used a per-call random draw.  The rollout
    config already exposes the same knobs as MixGRPO (`sample_strategy`,
    `iters_per_group`, `sde_window_seed`); honor them here so debug and
    training runs are reproducible and window placement is not coupled to the
    request UUID.
    """
    if window_size is None or int(window_size) <= 0:
        return None

    size = int(window_size)
    strategy = str(extra_args.get("sample_strategy", "random"))

    if strategy == "progressive":
        low = int(window_range[0]) if window_range is not None else 0
        high = int(window_range[1]) if window_range is not None else size
        max_start = high - size
        if max_start < low:
            return (low, low + size)

        global_steps = int(extra_args.get("global_steps", 0))
        iters_per_group = max(1, int(extra_args.get("iters_per_group", 1)))
        n_advances = max(0, global_steps) // iters_per_group
        begin = min(low + n_advances * size, max_start)
        return (begin, begin + size)

    if strategy == "random":
        sde_window_seed = extra_args.get("sde_window_seed")
        if sde_window_seed is not None:
            global_steps = int(extra_args.get("global_steps", 0))
            return _pick_sde_window(
                window_size=size,
                window_range=window_range,
                seed=int(sde_window_seed) + global_steps,
                request_id=None,
            )

        return _pick_sde_window(
            window_size=size,
            window_range=window_range,
            seed=seed,
            request_id=request_id,
        )

    logger.warning("Unknown BAGEL SDE sample_strategy=%r; falling back to per-request random window.", strategy)
    return _pick_sde_window(
        window_size=size,
        window_range=window_range,
        seed=seed,
        request_id=request_id,
    )


@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")
class BagelPipelineWithLogProb(BagelPipeline):
    """BAGEL pipeline variant for RL rollouts with verl-omni."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        inner = FlowMatchSDEDiscreteScheduler()
        self.scheduler = _BagelSchedulerAdapter(inner)
        logger.info("BagelPipelineWithLogProb: SDE scheduler enabled for RL rollouts")

    def _decode_token_prompt(self, token_ids: Any) -> str | None:
        """Decode BAGEL token IDs to a cleaned prompt text string."""
        token_list = _to_token_list(token_ids)
        if not token_list:
            return None
        decoded = self.tokenizer.decode(token_list, skip_special_tokens=False)
        return _extract_prompt_text(decoded)

    def _ensure_bagel_prompt_text(self, req: OmniDiffusionRequest) -> None:
        """Fill ``prompt`` and ``negative_prompt`` from token IDs if missing."""
        if not req.prompts or not isinstance(req.prompts[0], dict):
            return

        custom_prompt = req.prompts[0]
        if not custom_prompt.get("prompt"):
            prompt = self._decode_token_prompt(custom_prompt.get("prompt_token_ids"))
            if prompt is not None:
                custom_prompt["prompt"] = prompt

        extra_args = req.sampling_params.extra_args
        if "negative_prompt" not in extra_args:
            negative_prompt = self._decode_token_prompt(custom_prompt.get("negative_prompt_ids"))
            if negative_prompt is not None:
                extra_args["negative_prompt"] = negative_prompt

        prompt_extra_args = custom_prompt.get("extra_args")
        if isinstance(prompt_extra_args, dict):
            multi_modal_data = prompt_extra_args.get("multi_modal_data")
            if multi_modal_data is not None and "multi_modal_data" not in custom_prompt:
                custom_prompt["multi_modal_data"] = multi_modal_data

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        self._ensure_bagel_prompt_text(req)

        # Force trajectory recording on for RL
        req.sampling_params.return_trajectory_latents = True

        extra_args = req.sampling_params.extra_args

        # Apply CFG defaults so rollout and training log-prob recomputation match.
        for k, v in BAGEL_FLOWGRPO_CFG_DEFAULTS.items():
            extra_args.setdefault(k, v)
        if isinstance(extra_args.get("cfg_interval"), list):
            extra_args["cfg_interval"] = tuple(extra_args["cfg_interval"])

        # Pick SDE window: noise and log-prob recording only inside this range.
        logprobs = bool(extra_args.get("logprobs", True))
        noise_level = float(extra_args.get("noise_level", 0.0))
        sde_window_size = extra_args.get("sde_window_size", None)
        sde_window_range = extra_args.get("sde_window_range", None)
        if isinstance(sde_window_range, list):
            sde_window_range = tuple(sde_window_range)

        sde_window: Optional[tuple[int, int]] = None
        if sde_window_size and noise_level > 0.0:
            sde_window = _pick_strategy_sde_window(
                window_size=int(sde_window_size),
                window_range=sde_window_range,
                extra_args=extra_args,
                seed=req.sampling_params.seed,
                request_id=getattr(req, "request_id", None),
            )

        # Pass scheduler kwargs; _BagelSchedulerAdapter overrides noise_level
        # and return_logprobs per-step based on the SDE window.
        self.scheduler_kwargs = {k: extra_args[k] for k in ("noise_level", "sde_type", "generator") if k in extra_args}
        self.scheduler_kwargs["return_logprobs"] = logprobs
        # Official flow_grpo BAGEL stores only the quadratic term of the
        # Gaussian log-prob, omitting the normalizer constants.
        self.scheduler_kwargs["include_logprob_normalizer"] = False

        # Per-request scheduler setup matching training-side sigma schedule.
        assert req.sampling_params.num_inference_steps is not None, "num_inference_steps must be set for RL rollouts"
        bagel_num_timesteps = int(req.sampling_params.num_inference_steps)
        setup_bagel_sigmas(self.scheduler._inner, bagel_num_timesteps)

        # Reset adapter state *after* set_timesteps so inner step_index is None.
        self.scheduler.begin_forward(
            sde_window=sde_window,
            noise_level=noise_level,
            return_logprobs=logprobs,
        )

        # vllm-omni 0.22+ runs one extra denoise step vs official BAGEL; compensate
        # so rollout sigmas match setup_bagel_sigmas / training (issue #4470).
        req.sampling_params.num_inference_steps = vllm_omni_num_timesteps(bagel_num_timesteps)
        try:
            output = super().forward(req)
        finally:
            req.sampling_params.num_inference_steps = bagel_num_timesteps

        # Slice trajectory to the SDE window so training only sees noisy steps.
        traj_latents = output.trajectory_latents
        traj_timesteps = output.trajectory_timesteps
        traj_log_probs = output.trajectory_log_probs

        if sde_window is not None:
            begin, end = sde_window
            if traj_latents is not None:
                traj_latents = traj_latents[begin : end + 1]
            if traj_timesteps is not None:
                traj_timesteps = traj_timesteps[begin:end]

        _parity_dump(
            {
                "event": "rollout",
                "request_id": str(getattr(req, "request_id", "")),
                "sde_window": list(sde_window) if sde_window is not None else None,
                "timesteps": _parity_tensor_summary(traj_timesteps),
                "log_probs": _parity_tensor_summary(traj_log_probs),
                "first_latent": _parity_tensor_summary(traj_latents[0] if traj_latents is not None else None, limit=4),
                "last_latent": _parity_tensor_summary(traj_latents[-1] if traj_latents is not None else None, limit=4),
            }
        )

        return DiffusionOutput(
            output=maybe_to_cpu(output.output),
            custom_output={
                "all_latents": maybe_to_cpu(traj_latents.unsqueeze(0)) if traj_latents is not None else None,
                "all_timesteps": maybe_to_cpu(traj_timesteps.unsqueeze(0)) if traj_timesteps is not None else None,
                "all_log_probs": maybe_to_cpu(traj_log_probs.unsqueeze(0)) if traj_log_probs is not None else None,
            },
            trajectory_latents=None,
            trajectory_timesteps=None,
            trajectory_log_probs=None,
        )
