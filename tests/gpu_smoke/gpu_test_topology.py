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
import os

import torch


def resolve_requested_num_gpus(default_num_gpus: int = 4) -> int:
    """Resolve requested GPU count from NUM_GPUS, clamped by visible CUDA devices."""
    requested_gpus = int(os.getenv("NUM_GPUS", str(default_num_gpus)))
    available_gpus = torch.cuda.device_count()
    if available_gpus > 0:
        requested_gpus = min(requested_gpus, available_gpus)
    return max(1, requested_gpus)


def resolve_diffusion_gpu_topology(default_num_gpus: int = 4) -> tuple[int, int, int]:
    """Return (requested_gpus, tensor_parallel_size, attention_heads)."""
    requested_gpus = resolve_requested_num_gpus(default_num_gpus=default_num_gpus)

    # The tiny Qwen-Image test model in smoke tests supports TP=1 or TP=2.
    tensor_parallel_size = 2 if requested_gpus >= 2 else 1
    attention_heads = 2 if tensor_parallel_size == 2 else 1
    return requested_gpus, tensor_parallel_size, attention_heads


def resolve_reward_model_gpu_topology(default_num_gpus: int = 4) -> tuple[int, int]:
    """Return (reward_model_gpus, tensor_parallel_size) for reward-model tests."""
    requested_gpus = resolve_requested_num_gpus(default_num_gpus=default_num_gpus)

    # Keep TP conservative for broad compatibility in smoke runs.
    reward_model_gpus = min(2, requested_gpus)
    tensor_parallel_size = 2 if reward_model_gpus >= 2 else 1
    return reward_model_gpus, tensor_parallel_size
