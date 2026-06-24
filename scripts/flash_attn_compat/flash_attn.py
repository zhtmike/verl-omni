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

"""Minimal FlashAttention compatibility shim for BAGEL debug probes.

This module is intentionally small and only covers the call patterns used by
the official flow_grpo BAGEL model.  It lets the one-step parity probe run in an
environment without the compiled flash-attn package by delegating to PyTorch
scaled dot-product attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_kv_for_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if q_heads == kv_heads:
        return k, v
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
    repeat = q_heads // kv_heads
    return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    # FlashAttention aligns causal masks to the bottom-right when q_len != k_len,
    # which is what KV-cache decoding expects.
    q_pos = torch.arange(q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    return k_pos <= (k_len - q_len + q_pos)


def _sdpa_segment(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    q_b = q.transpose(0, 1).unsqueeze(0)  # (1, H, Lq, D)
    k_b = k.transpose(0, 1).unsqueeze(0)  # (1, H, Lk, D)
    v_b = v.transpose(0, 1).unsqueeze(0)
    # Prefer fused SDPA for the common full-sequence causal case.  A materialized
    # mask here can force the memory-heavy math backend and OOM on 512px BAGEL.
    use_builtin_causal = causal and q.shape[0] == k.shape[0]
    attn_mask = _causal_mask(q.shape[0], k.shape[0], q.device) if causal and not use_builtin_causal else None
    try:
        out = F.scaled_dot_product_attention(
            q_b,
            k_b,
            v_b,
            attn_mask=attn_mask,
            is_causal=use_builtin_causal,
            enable_gqa=q.shape[1] != k.shape[1],
        )
    except TypeError:
        k, v = _expand_kv_for_gqa(q, k, v)
        k_b = k.transpose(0, 1).unsqueeze(0)
        v_b = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask, is_causal=use_builtin_causal)
    return out.squeeze(0).transpose(0, 1)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,  # noqa: ARG001
    max_seqlen_k: int | None = None,  # noqa: ARG001
    dropout_p: float = 0.0,
    softmax_scale=None,  # noqa: ANN001, ARG001
    causal: bool = False,
    **kwargs,  # noqa: ANN003, ARG001
) -> torch.Tensor:
    if dropout_p != 0.0:
        raise NotImplementedError("debug flash_attn shim only supports dropout_p=0")

    outputs: list[torch.Tensor] = []
    cu_q = cu_seqlens_q.to(device="cpu", dtype=torch.long).tolist()
    cu_k = cu_seqlens_k.to(device="cpu", dtype=torch.long).tolist()
    for i in range(len(cu_q) - 1):
        q_i = q[cu_q[i] : cu_q[i + 1]]
        k_i = k[cu_k[i] : cu_k[i + 1]]
        v_i = v[cu_k[i] : cu_k[i + 1]]
        outputs.append(_sdpa_segment(q_i, k_i, v_i, causal=causal))
    return torch.cat(outputs, dim=0)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale=None,  # noqa: ANN001, ARG001
    causal: bool = False,
    **kwargs,  # noqa: ANN003, ARG001
) -> torch.Tensor:
    if dropout_p != 0.0:
        raise NotImplementedError("debug flash_attn shim only supports dropout_p=0")

    if q.ndim == 3:
        return _sdpa_segment(q, k, v, causal=causal)
    if q.ndim != 4:
        raise ValueError(f"expected q with 3 or 4 dims, got {q.shape}")

    outputs = []
    for i in range(q.shape[0]):
        outputs.append(_sdpa_segment(q[i], k[i], v[i], causal=causal))
    return torch.stack(outputs, dim=0)
