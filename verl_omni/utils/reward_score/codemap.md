# verl_omni/utils/reward_score/

## Responsibility
Visual/audio reward scoring functions for generative (image/video/audio) RL training: rule-based, model-based (local checkpoints), and remote HTTP/LLM-judge scorers that map a generated sample plus prompt/ground truth to a reward.

## Design / Registration mechanism
There is no central decorator registry. Dispatch is `data_source`-string based: `reward_score/__init__.default_compute_score_image(data_source, solution_image, ground_truth, extra_info)` routes to a scorer with an if-branch per registered name (currently only `jpeg_compressibility`; anything else raises `NotImplementedError`). The `RewardManagerBase`-based `verl_omni/reward_loop/reward_manager/visual.py` (`VisualRewardManager`) reads `data_source` from `non_tensor_batch` and calls either the injected `compute_score` or `default_compute_score_image`, passing `reward_router_address` / `reward_model_tokenizer` / `model_name` kwargs to async scorers. Task-specific dispatch tables live in per-task `compute_score` functions (e.g. `examples/diffusionopd_trainer/sd35/mixed_task_reward.py`, `examples/gspo_trainer/data_process/mmk12.py`) that import these scorers directly.

## Flow
1. Reward manager collects generated samples (`rm_scores` shaped `(batch, 1)`); per sample it extracts `data_source`, `ground_truth`, and the rollout image tensor.
2. `default_compute_score_image` (or the config-injected custom scorer) dispatches on `data_source` to a scorer below.
3. Model-based scorers lazily build a singleton inferencer, batch concurrent requests via an asyncio consumer loop, and return `float` or `{"score": ..., "reward_extra_info": {...}}`.

## Integration
- Consumed by: `verl_omni/reward_loop/reward_manager/visual.py`, example per-task reward configs, tests under `tests/utils/reward_score/`.
- Depends on: torch, PIL, transformers (Qwen2VL for HPSv3), `transformers ClapModel` / ImageBind checkpoints, vLLM OpenAI-compatible router (`reward_router_address`) for LLM judges, aiohttp for HTTP scorers.
- Key entry points: `default_compute_score_image` plus per-scorer `compute_score*` functions.

## Scorer inventory
- `jpeg_compressibility.py` — `compute_score` (+ `jpeg_compressibility()` / `jpeg_incompressibility()` factories): differentiable-ish JPEG size proxy reward; the only scorer registered in `default_compute_score_image`.
- `pickscore_reward.py` — `compute_score_pickscore`: PickScore CLIP-based human-preference scoring; `_PickScoreInferencer` + `_consumer_loop`/`_ensure_consumer` async batching.
- `hpsv3_reward.py` — `compute_score_hpsv3`: HPSv3 (Qwen2VL-based, `_Qwen2VLRewardModelBT`) preference score with smart image resizing (`_smart_resize`) and video frame extraction (`_extract_frames`); `_HPSv3Inferencer` cached via `_get_inferencer`.
- `unified_reward.py` — `compute_score_unified_reward`: LLM-judge (UnifiedReward 2.0) via vLLM router; scores Alignment/Coherence/Style per frame (`_score_single_image`, `_parse_unified_reward_scores`, `_aggregate_unified_reward_scores`), normalizes 1–5 → [0,1].
- `genrm_ocr.py` — `compute_score_ocr`: GenRM OCR reward — LLM-judge transcription scored by normalized Levenshtein (`_levenshtein_score`) against ground truth.
- `clap.py` — `compute_score`: CLIP-style CLAP audio–text similarity; `_get_audio` pulls waveform from `extra_info`, `_BatchingState`/`_consumer_loop` batch scoring.
- `imagebind.py` — `compute_score`: ImageBind embeddings over audio/video/text with cosine similarity and weighted aggregation (`_aggregate_similarities`); audio preprocessed via mel spectrogram (`_waveform_to_melspec`).
- `mmk12_reward.py` — `compute_score`: rule-based multiple-choice/format scoring for math-style answers (`_extract_choice_letter`, `_compute_format_score`).
- `choice_reward.py` — `compute_score`: simple choice-extraction exact-match reward (`extract_answer`).
- `http_scorer_client.py` — `compute_score`: POSTs serialized PIL image bytes to a remote HTTP scorer service.
- `latent_http_scorer_client.py` — `compute_score`: sends raw latent tensors (`_serialize_request`) to a remote latent-space scorer over aiohttp.
- `reward_utils.py` — shared helpers: `video_tensor_to_pil_frames`, `pil_image_to_base64`.
