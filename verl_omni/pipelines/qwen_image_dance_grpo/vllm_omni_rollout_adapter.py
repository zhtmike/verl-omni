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
"""Qwen-Image rollout adapter for the DanceGRPO algorithm.

The rollout-side SDE loop is structurally identical to the FlowGRPO loop:
both encode prompts, iterate the scheduler under SDE noise, and ship
``all_latents`` / ``all_log_probs`` / ``all_timesteps`` together with the
prompt embeddings via :class:`vllm_omni.diffusion.data.DiffusionOutput`.

The DanceGRPO loop differs in two places:

1.  All denoising steps inject SDE noise (the original implementation has
    no concept of a contiguous "SDE window" — every step contributes a
    log-probability), so we force ``sde_window`` to span the full
    trajectory.
2.  The scheduler step uses ``sde_type="dance"`` (see
    :class:`~verl_omni.pipelines.schedulers.FlowMatchSDEDiscreteScheduler`).

Everything else (prompt encoding, CFG, VAE decode) is reused from the
FlowGRPO Qwen-Image rollout via subclassing.
"""

from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import (
    QwenImagePipelineWithLogProb,
)

__all__ = ["QwenImageDanceGRPOPipelineWithLogProb"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dance_grpo")
class QwenImageDanceGRPOPipelineWithLogProb(QwenImagePipelineWithLogProb):
    """Rollout pipeline for Qwen-Image under DanceGRPO.

    Subclasses the FlowGRPO Qwen-Image rollout pipeline and pins the SDE
    variant to ``"dance"`` while disabling the SDE-window mechanism (every
    step injects noise).
    """

    def forward(  # type: ignore[override]
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        max_sequence_length: int = 512,
        noise_level: float = 0.3,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 1),
        sde_type: Literal["sde", "cps", "dance"] = "dance",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        # Pin the algorithm-specific knobs and otherwise delegate to the
        # FlowGRPO loop. The base class respects ``sampling_params.extra_args``
        # so callers can still override ``noise_level`` per-request.
        return super().forward(
            req=req,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_mask=negative_prompt_mask,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            output_type=output_type,
            attention_kwargs=attention_kwargs,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            noise_level=noise_level,
            # ``None`` instructs ``diffuse`` to span every denoising step,
            # matching the DanceGRPO reference implementation in
            # ``DanceGRPO/fastvideo/train_grpo_qwenimage.py``.
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range,
            sde_type=sde_type,
            logprobs=logprobs,
        )
