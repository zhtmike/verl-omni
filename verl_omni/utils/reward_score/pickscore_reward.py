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

"""PickScore reward function for FlowGRPO training.

PickScore uses a fine-tuned CLIP-ViT-H model to score image-text alignment.
This is the reward used by the official BAGEL FlowGRPO ``pickscore_bagel_lora``
config in the flow_grpo repo.

Reference: https://github.com/yuvalkirstain/PickScore
"""

import logging
import threading
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_scorer = None


def _get_scorer():
    """Lazily load the PickScore model on GPU (matching official flow_grpo).

    PickScore is ~1GB (CLIP-ViT-H).  Official flow_grpo runs it on GPU via a
    thread-pool executor.  The reward computation happens asynchronously so it
    does not block the actor rollout.
    """
    global _scorer
    if _scorer is None:
        with _lock:
            if _scorer is None:
                from transformers import CLIPModel, CLIPProcessor

                processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                model_path = "yuvalkirstain/PickScore_v1"

                device = "cuda" if torch.cuda.is_available() else "cpu"
                processor = CLIPProcessor.from_pretrained(processor_path)
                model = CLIPModel.from_pretrained(model_path).eval().to(device)

                _scorer = (processor, model, device)
                logger.info("PickScore model loaded: %s on %s", model_path, device)
    return _scorer


def _to_pil(image: torch.Tensor | np.ndarray | Image.Image) -> Image.Image:
    """Normalize a tensor / array / PIL image to a uint8 RGB PIL image."""
    if isinstance(image, torch.Tensor):
        image = image.float().permute(1, 2, 0).cpu().numpy()
    if isinstance(image, np.ndarray):
        # Handle [C, H, W] or [H, W, C]
        if image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0) if image.shape[0] == 3 else image
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)
    return image


def compute_score(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer=None,
    model_name: Optional[str] = None,
    **kwargs,
) -> dict:
    """Compute PickScore for an image given a text prompt.

    Args:
        data_source: Source dataset identifier (unused).
        solution_image: Generated image as tensor [C, H, W] or [N, C, H, W].
        ground_truth: The text prompt used to generate the image.
        extra_info: Additional metadata (unused).
        reward_router_address: (unused, kept for interface consistency).
        reward_model_tokenizer: (unused, kept for interface consistency).
        model_name: (unused, kept for interface consistency).

    Returns:
        dict: ``{"score": float}`` where score is in approximately [-1, 1] range
        (CLIP cosine similarity scaled by logit_scale / 26).
    """
    # Handle single image [C, H, W] → [1, C, H, W]
    if isinstance(solution_image, torch.Tensor) and solution_image.ndim == 3:
        solution_image = solution_image.unsqueeze(0)
    elif isinstance(solution_image, np.ndarray) and solution_image.ndim == 3:
        solution_image = solution_image[np.newaxis, ...]

    # Convert to list of PIL images
    if isinstance(solution_image, torch.Tensor):
        pil_images = [_to_pil(img) for img in solution_image]
    else:
        pil_images = [_to_pil(img) for img in solution_image]

    prompt_text = str(ground_truth) if ground_truth else ""

    processor, model, device = _get_scorer()

    # Preprocess images
    image_inputs = processor(
        images=pil_images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

    # Preprocess text (repeat for multiple images)
    prompts = [prompt_text] * len(pil_images)
    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        image_outputs = model.get_image_features(**image_inputs)
        image_embs = image_outputs.pooler_output if hasattr(image_outputs, "pooler_output") else image_outputs
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_outputs = model.get_text_features(**text_inputs)
        text_embs = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag() / 26.0  # normalize to ~[0, 1]

    # Average over multiple images if provided
    avg_score = float(scores.mean().cpu().item())

    return {"score": avg_score}
