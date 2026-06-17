# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

"""
Preprocess the PickScore dataset for BAGEL FlowGRPO training.

Prompts are stored in standard chat-message format.  The BAGEL tokenizer
(used by the agent loop and training adapter) produces the correct
BAGEL-format token IDs automatically via the standard ``prompts`` tensor.

The official PickScore dataset is available from the flow_grpo repository.
To prepare it::

    wget -P ~/data/pickscore/ \
      https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/train.txt
    wget -P ~/data/pickscore/ \
      https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/test.txt
    python examples/flowgrpo_trainer/data_process/bagel_pickscore.py

If you have your own prompt dataset, place train.txt / test.txt in any
directory and pass ``--input_dir`` / ``--output_dir``.
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PickScore dataset for BAGEL FlowGRPO training.")
    parser.add_argument(
        "--input_dir",
        default="~/data/pickscore/",
        help="Directory containing train.txt and test.txt (one caption per line).",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/pickscore/bagel",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS output directory.")

    args = parser.parse_args()

    local_dataset_path = os.path.expanduser(args.input_dir)

    train_file = os.path.join(local_dataset_path, "train.txt")
    test_file = os.path.join(local_dataset_path, "test.txt")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Expected raw text files at {train_file} and {test_file}. "
            f"Download them from the flow_grpo repository:\n"
            f"  wget -P {local_dataset_path} \\\n"
            f"    https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/train.txt\n"
            f"  wget -P {local_dataset_path} \\\n"
            f"    https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/test.txt"
        )
    dataset = datasets.load_dataset("text", data_files={"train": train_file, "test": test_file})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    data_source = "flow_grpo/pickscore"

    def make_map_fn(split: str):
        def process_fn(example, idx):
            prompt_text = example.pop("text").strip()
            if not prompt_text:
                return None

            # PickScore compares the generated image against the prompt text.
            # Set ground_truth = prompt_text directly.
            return {
                "data_source": data_source,
                "prompt": [
                    {"role": "user", "content": prompt_text},
                ],
                "negative_prompt": [
                    {"role": "user", "content": " "},
                ],
                "ability": "pickscore",
                "reward_model": {
                    "style": "model",
                    "ground_truth": prompt_text,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Saved preprocessed PickScore dataset to {local_save_dir}/")
    print(f"  train: {len(train_dataset)} samples")
    print(f"  test:  {len(test_dataset)} samples")

    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir, exist_ok=True)
            copy(src=local_save_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: verl not installed, skipping HDFS upload.")
