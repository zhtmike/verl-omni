# verl_omni/utils/dataset/visual_reflection/

## Responsibility
Stable public API for building generic multi-turn visual-reflection SFT data: schema contracts, source-record adapters, atomic-example planning, dedup/split partitioning, image asset resolution, reflection response protocol, and full provenance manifests with rejection ledgers.

## Design
TypedDict-driven contracts (`contracts.py`) with strict validators and stable digests; pipeline stages are pure functions over those contracts. `__init__.py` re-exports a curated `__all__` (e.g. `VisualReflectionTrajectory`, `ReflectionStep`, `plan_atomic_examples`, `validate_trajectory`, `RejectionLedger`, `build_data_manifest`, `format_reflection_response`, `parse_reflection_response`). All errors surface as `VisualReflectionDataError` with machine-readable `RejectionReason`s.

## Flow
1. Source records (Echo pair, Midjourney prompt, UniCoT) are adapted via `sources.adapt_echo_pair_record` / `adapt_midjourney_prompt_record` / `unicot.parse_unicot_record` into validated synthesis/verification request TypedDicts (`build_pair_synthesis_request`, `make_draft_generation_request`, `make_reflection_synthesis_request`, ...).
2. `partition.deduplicate_source_records` + `assign_source_splits` (seeded, deterministic `derive_split_for_dedup_key`) partition records; `images.LocalImageResolver` (implements `ImageResolver` protocol) resolves/validates image assets with sha256 checks.
3. `planner.plan_atomic_examples` / `validate_atomic_example` fold trajectories into atomic SFT examples (`T2IAtom`, `ReflectAtom`, `EditAtom`); `protocol.parse_reflection_response` / `format_reflection_response` encode/decode the reflection turn format.
4. `provenance.RejectionLedger` records every rejection/error as JSONL; `build_data_manifest` + `derive_manifest_id` emit a fully auditable `DataManifest` with source/model/split provenance and counts.

## Integration
- Consumed by: offline data-conversion pipelines producing visual-reflection SFT parquets consumed by the DPO/SFT trainers.
- Depends on: stdlib only (dataclasses, typing, json, hashlib); no torch/verl dependency — dataset construction is offline and side-effect free.
- Key entry points: `contracts.validate_trajectory`, `planner.plan_atomic_examples`, `partition.assign_source_splits`, `images.LocalImageResolver`, `protocol.parse_reflection_response`, `provenance.build_data_manifest`.
- Files: `contracts.py` (schemas/validators/dedup keys), `sources.py` (record adapters + request/result builders/validators), `unicot.py` (UniCoT record parser, `unicot_converter_config`), `partition.py` (dedup + deterministic splits), `images.py` (asset resolution), `protocol.py` (reflection wire format), `planner.py` (atomic example planning), `provenance.py` (manifests, rejection ledger).
