# verl_omni/utils/dataset/

## Responsibility
Dataset classes and transforms for RL and offline DPO training: parquet-backed RLHF datasets extended with diffusion negative-prompt and omni-modal (image/audio/video) support, plus offline preference datasets that tokenize chosen/rejected branches into model-ready tensors.

## Design
Thin subclasses of upstream veRL datasets (`RLHFDataset`, `MultiTurnSFTDataset`) plus factory functions selected from `DictConfig` (`get_dataset_class`, `get_collate_fn`, `create_rl_dataset`, `create_rl_sampler`, which support `custom_cls.path` external loading). Multimodal loading is pluggable via `_process_multi_modal_info` classmethod overrides.

## Flow
1. Trainer calls `create_rl_dataset(data_paths, data_config, tokenizer, processor)` → `get_dataset_class` picks `RLHFDataset` (verl-omni) or `QwenOmniRLHFDataset` for omni-modal data; `create_rl_sampler` builds the batch sampler.
2. `RLHFDataset.__getitem__` reads a parquet row, builds `raw_prompt` (apply_chat_template deferred to AgentLoop) and, for diffusion CFG training, exposes `raw_negative_prompt` from `negative_prompt_key`.
3. Offline DPO: `OfflineDPODataset.__getitem__` tokenizes prompt + chosen/rejected responses (`_tokenize_prompt`, `_score_from_column`); `expand_offline_dpo_features` + `offline_dpo_collate_fn` batch them.
4. MLLM DPO: `OfflineMLLMDPODataset._read_files_and_process` builds preference branches (`_process_preference_branch`, `_pair_branch_values`); `offline_mllm_dpo_collate_fn` pads tensors (pad-mode aware) and `ModalityGroupedBatchSampler` yields modality-balanced batches.

## Integration
- Consumed by: trainer entrypoints and DPO trainers under `verl_omni/trainer/`, reward-free offline pipelines.
- Depends on: `verl.utils.dataset.rl_dataset.RLHFDataset`, `qwen_omni_utils.process_mm_info`, transformers processors, pandas/pyarrow parquet loading.
- Key entry points: `RLHFDataset`, `QwenOmniRLHFDataset`, `OfflineDPODataset`, `OfflineMLLMDPODataset`, `ModalityGroupedBatchSampler`, `create_rl_dataset`, `create_rl_sampler`, `offline_dpo_collate_fn`, `offline_mllm_dpo_collate_fn`, `process_qwen3_omni_sample`.
- Files:
  - `rl_dataset.py` — RLHFDataset with `negative_prompt` support + dataset/collate/sampler factories.
  - `omni_rl_datasets.py` — `QwenOmniRLHFDataset`; overrides `_process_multi_modal_info` to use Qwen `process_mm_info`, returns `(images, videos, audios)`, and zero-pads audio via `pad_audio_to_hop_multiple` (hop 160) so HF and vllm-omni expand identical audio token counts.
  - `offline_dpo_dataset.py` — text-only offline DPO parquet dataset.
  - `offline_mllm_dpo_dataset.py` — multimodal (image/video/audio) offline DPO dataset + modality-grouped sampler.
  - `qwen3_omni_transform.py` — `process_qwen3_omni_sample`: renders Qwen3-Omni chat template, decodes images/videos/audios (`_fetch_images/_fetch_videos/_fetch_audios`, smart resize, frame-index calculation), builds input_ids, labels with `IGNORE_INDEX`/modality token offsets (`IMAGE_INPUT_INDEX=-200`, `VIDEO_INPUT_INDEX=-300`, `AUDIO_INPUT_INDEX=-400`), position ids, and assistant token masks.
