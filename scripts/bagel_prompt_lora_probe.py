#!/usr/bin/env python3
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

"""Lightweight BAGEL FlowGRPO prompt and LoRA parity probe."""

from __future__ import annotations

import os
from pathlib import Path

import datasets
from transformers import AutoTokenizer

from examples.flowgrpo_trainer.data_process.bagel_pickscore import bagel_prepare_prompt_token_ids
from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import _extract_prompt_text


def _load_prompt(dataset_path: str) -> tuple[str, list[int]]:
    ds = datasets.load_dataset("parquet", data_files={"train": dataset_path}, split="train")
    row = ds[0]
    prompt_text = row["prompt"][0]["content"]
    return prompt_text, [int(x) for x in row["prompt_token_ids"]]


def main() -> None:
    model_path = os.path.expanduser("~/models/ByteDance-Seed/BAGEL-7B-MoT")
    dataset_path = os.path.expanduser("~/data/pickscore/bagel/train.parquet")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    bos = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")

    prompt, stored_ids = _load_prompt(dataset_path)
    local_ids = bagel_prepare_prompt_token_ids(tokenizer, prompt)
    vllm_ids = [bos, *tokenizer.encode(prompt.strip(), add_special_tokens=False), eos]
    official_ids = [bos, *tokenizer.encode(prompt.strip()), eos]

    decoded = tokenizer.decode(stored_ids, skip_special_tokens=False)
    extracted = _extract_prompt_text(decoded)
    roundtrip_ids = [bos, *tokenizer.encode(extracted, add_special_tokens=False), eos]

    print("prompt_text", prompt)
    print("prompt_len_chars", len(prompt))
    print("stored_len", len(stored_ids))
    print("local_helper_match", stored_ids == local_ids)
    print("vllm_prepare_match", stored_ids == vllm_ids)
    print("official_prepare_match", stored_ids == official_ids)
    print("decode_extract_match", extracted == prompt.strip())
    print("roundtrip_match", stored_ids == roundtrip_ids)
    print("stored_first_last", stored_ids[0], stored_ids[-1])
    print("bos_eos", bos, eos)

    official_targets = [
        "self_attn.q_proj_moe_gen",
        "self_attn.k_proj_moe_gen",
        "self_attn.v_proj_moe_gen",
        "self_attn.o_proj_moe_gen",
        "mlp_moe_gen.gate_proj",
        "mlp_moe_gen.up_proj",
        "mlp_moe_gen.down_proj",
    ]
    verl_targets = [
        "q_proj_moe_gen",
        "k_proj_moe_gen",
        "v_proj_moe_gen",
        "o_proj_moe_gen",
        "mlp_moe_gen.gate_proj",
        "mlp_moe_gen.up_proj",
        "mlp_moe_gen.down_proj",
    ]

    from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration

    mapping = OmniBagelForConditionalGeneration.packed_modules_mapping
    print("official_lora_targets", official_targets)
    print("verl_lora_targets", verl_targets)
    print("packed_qkv_moe_gen", mapping.get("qkv_proj_moe_gen"))
    print("packed_gate_up_moe_gen", mapping.get("mlp_moe_gen.gate_up_proj"))
    print("expected_lora_rank_alpha", 64, 128)
    print("local_config_lora_dtype", "float32")
    print("official_cast_lora_dtype", "bf16")
    print("dataset_path_exists", Path(dataset_path).exists())


if __name__ == "__main__":
    main()
