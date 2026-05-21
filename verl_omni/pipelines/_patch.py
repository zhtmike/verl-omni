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
import logging
import os

import diffusers
import torch
from packaging import version

logger = logging.getLogger(__name__)


def _apply_qwen_image_ulysses_mask_fix() -> None:
    if version.parse(diffusers.__version__) < version.parse("0.38.0"):
        return

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    _orig_forward = QwenImageTransformer2DModel.forward
    if getattr(_orig_forward, "_verl_omni_ulysses_mask_patched", False):
        return

    def _patched_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_kwargs=None,
        **kwargs,
    ):
        parallel_config = getattr(self, "_parallel_config", None)
        cp_config = parallel_config.context_parallel_config if parallel_config is not None else None
        ulysses_degree = cp_config.ulysses_degree if cp_config is not None else 1

        if ulysses_degree > 1 and encoder_hidden_states_mask is not None:
            if not _patched_forward._warned:
                logger.warning(
                    "verl_omni patch applied: QwenImageTransformer2DModel.forward has been monkey-patched to fix "
                    "the Ulysses SP joint-attention-mask layout bug (diffusers==0.38). "
                    "The joint mask is now built in interleaved [txt_0, img_0, txt_1, img_1, ...] order "
                    "to match the post-all-to-all sequence layout when ulysses_degree > 1. "
                    "Remove this patch once the fix is upstreamed to diffusers."
                )
                _patched_forward._warned = True
            # Build the joint mask in the interleaved layout that matches the
            # post-all-to-all sequence order: [txt_0, img_0, txt_1, img_1, ...]
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            txt_chunks = encoder_hidden_states_mask.chunk(ulysses_degree, dim=1)
            img_chunks = image_mask.chunk(ulysses_degree, dim=1)
            joint_mask = torch.cat([x for pair in zip(txt_chunks, img_chunks, strict=False) for x in pair], dim=1)
            attention_kwargs = dict(attention_kwargs or {}, attention_mask=joint_mask[:, None, None, :])
            encoder_hidden_states_mask = None

        return _orig_forward(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            attention_kwargs=attention_kwargs,
            **kwargs,
        )

    _patched_forward._verl_omni_ulysses_mask_patched = True
    _patched_forward._warned = False
    QwenImageTransformer2DModel.forward = _patched_forward


def _apply_diffusion_master_port_env_fix() -> None:
    """Honor MASTER_PORT set by verl-omni rollout servers for diffusion workers.

    vllm-omni's OmniDiffusionConfig.__post_init__ picks a random port in a narrow
    range (30005 + rand(0, 100)), which collides when multiple Ray vLLMOmniHttpServer
    actors start in parallel. verl's vLLM reward path avoids this via get_free_port().

    # TODO (long): drop this
    """
    try:
        from vllm_omni.diffusion.data import OmniDiffusionConfig
    except ImportError:
        return

    if getattr(OmniDiffusionConfig, "_verl_omni_master_port_patched", False):
        return

    _orig_post_init = OmniDiffusionConfig.__post_init__

    def _patched_post_init(self) -> None:
        env_port = os.environ.get("MASTER_PORT")
        reserved_port: int | None = None
        if env_port is not None:
            try:
                reserved_port = int(env_port)
            except ValueError:
                logger.warning("Ignoring invalid MASTER_PORT=%r for diffusion workers", env_port)

        _orig_post_init(self)

        if reserved_port is not None:
            if not _patched_post_init._warned:
                logger.warning(
                    "verl_omni patch applied: OmniDiffusionConfig.__post_init__ now honors "
                    "MASTER_PORT from the environment (set by vLLMOmniHttpServer) instead of "
                    "vllm-omni's random port selection. Remove this patch once vllm-omni "
                    "respects an explicit master_port without adding random jitter."
                )
                _patched_post_init._warned = True
            self.master_port = reserved_port

    _patched_post_init._warned = False
    OmniDiffusionConfig.__post_init__ = _patched_post_init
    OmniDiffusionConfig._verl_omni_master_port_patched = True


_apply_qwen_image_ulysses_mask_fix()
_apply_diffusion_master_port_env_fix()
