# Identity-aware Seed Bank

This is a bounded development tool, separate from frozen full-corpus outputs.
It does not modify accepted labels or automatically promote masks to GT.

## Environment

Reuse `<ANNOTATION_ENV>`.
Base installation is in README.md and SAM installation in MASKS.md.
The current server already has official GroundingDINO Swin-T, SAM2.1 base+,
and local frozen DINOv2 ViT-S/14 (`third_party/dinov2`,
`models/dinov2/dinov2_vits14_pretrain.pth`). No network download or training
is needed. The model hash and code hashes are in build_report.json/run_config.json.
GPU1 is excluded from these experiments. Review HTML needs no Python or GPU.

## Reproduce into NEW output directories

```bash
cd <SOURCE_WORKSPACE>
conda activate astribot-annotation
CUDA_VISIBLE_DEVICES=2 python tools/object_annotation/seed_bank.py --output annotations/seed_bank_new
CUDA_VISIBLE_DEVICES=2 python tools/object_annotation/seed_bank_pilot.py --bank annotations/seed_bank_new --output annotations/seed_pilot_new
```

The dated assistant review config is tied to the exact original bank hash.
Do not apply it to an independently rebuilt bank with different provenance.
Existing curated bank was created with:

```bash
python tools/object_annotation/curate_seed_bank.py --bank annotations/identity_seed_bank_20260918 --review tools/object_annotation/config/seed_bank_visual_review_20260918.json --output annotations/identity_seed_bank_20260918_curated
CUDA_VISIBLE_DEVICES=2 python tools/object_annotation/rescore_seed_bank_pilot.py --bank annotations/identity_seed_bank_20260918_curated --parent annotations/identity_seed_bank_pilot_20260918 --output annotations/identity_seed_bank_pilot_20260918_curated
python tools/object_annotation/validate_seed_bank.py --bank annotations/identity_seed_bank_20260918_curated --pilot annotations/identity_seed_bank_pilot_20260918_curated
python tools/object_annotation/seed_bank_tables.py --pilot annotations/identity_seed_bank_pilot_20260918_curated
python tools/object_annotation/seed_bank_review.py --bank annotations/identity_seed_bank_20260918_curated
```

Creation commands refuse existing output directories. Existing completed artifacts
can be validated/read directly; do not rerun creation over them.

## Candidate integration API

`seed_bank_pilot.py` demonstrates the complete detector -> SAM image mask ->
frozen embedding -> ranking path. Production can call the same small API:

```python
from identity_guard import Appearance
from seed_bank import SeedBank, crops, rank_score

bank = SeedBank(bank_path)
encoder = Appearance()  # initialize once, not once per frame
rgb_crop, masked_crop, crop_mask, crop_box = crops(bgr_image, binary_mask)
rgb = encoder.embed([rgb_crop])[0].cpu().numpy()
masked = encoder.embed([masked_crop])[0].cpu().numpy()
match = bank.query(rgb, masked, object_id, camera,
                   exclude_episode=episode_id, mode="masked")
ranking_score = rank_score(detector_score, match)
```

Generate several candidates for only the current task's identities. Embed them
in a batch as in the pilot, retain both detector_score and retrieval details,
and sort each target's candidate list. Do not assume a top-ranked candidate is
present or valid. The example query excludes same-episode references for the
development comparison; within-episode **accepted** keyframes may deliberately
serve as seeds in a separately recorded recovery run, not an independent test.

Each bank entry retains object_id, robot_part_id, identity_name, camera,
episode_id, frame_idx, timestamp, bbox_xyxy, original-resolution COCO RLE,
full/crop PNG masks, crop geometry, provenance, tags, unit embedding index,
and bank_status. Excluded exemplars remain traceable but never match queries.
`left_ee/right_ee` source IDs1103/1104 have display aliases
`left_gripper/right_gripper`; source labels are unchanged. Per-sample
merged_object_ids are honored, not replaced by a global meat-class merge.

Same-camera reference is preferred; fallback, missing reference, competitor,
and nearest IDs are explicit. Cosines/margins are not calibrated confidence.
Robot laterality must not be inferred from appearance alone or from image-left.

## Offline review

Extract the download ZIP and open
`annotations/identity_seed_bank_20260918_curated/review.html` locally.
No SSH tunnel, HTTP server, account or external assets are needed.
RGB and masked crops are shown together. Clicking either opens full crop size;
original-resolution binary masks are also included under exemplars/.
Use identity/camera/status filters. `Accept Visible` accepts the current
include/exclude decisions for the filtered **pending** rows, not every label.
Explicit include/exclude/uncertain overrides and notes remain editable.
Export Review downloads a hash-bound JSON for return to the server.
Local browser storage is a convenience only: export before moving the folder
or changing browsers. Import rejects a mismatched bank hash. No review edits
change immutable bank.json until a later explicit versioned import.

The assistant reviewed all265 exemplar pairs and excluded24 unsuitable
retrieval references. The user accepted all curated decisions on2026-09-18.
Acceptance is a versioned sidecar under identity_seed_bank_transfer_20260918_r2;
immutable bank.json and its24 retrieval exclusions remain unchanged.

Full findings and limitations: docs/identity_seed_bank_pilot_report_20260918.md.

## Downstream transfer experiment

The transfer test does not demonstrate aggregate propagation improvement.
See docs/identity_seed_bank_transfer_report_20260918.md before promoting it.
57 unit tests cover selection, string robot IDs and QA video geometry.
Reproduce into new paths (never overwrite completed output):

```bash
python tools/object_annotation/seed_bank_transfer.py prepare --output annotations/transfer_new --eligible-first
CUDA_VISIBLE_DEVICES=2 python tools/object_annotation/seed_bank_transfer.py run --output annotations/transfer_new
python tools/object_annotation/seed_bank_transfer.py score --output annotations/transfer_new
python tools/object_annotation/seed_bank_transfer_qa.py --root annotations/transfer_new
python tools/object_annotation/validate_seed_bank_transfer.py --root annotations/transfer_new
```

Optional --shard N --shards K assigns disjoint cases to independent workers.
Prepare --scoring-only --candidate-parent PATH reuses hashed candidate masks
from a completed parent experiment. --eligible-first filters detector<0.30
before ranking, so an ineligible top-ranked candidate cannot discard the rest.
The historical r2 config intentionally records the old order. Use only
qa_verified/ videos; original r2 previews had an incompatible legacy geometry.

## Admission and assisted-seed ablation

`seed_admission_pilot.py` reuses the frozen recovery candidate/check cache.
It compares guarded replacement, guarded initialization+replacement, and
explicit assistant point/box SAM2 seeds, with unchanged single-bank and
recovery-bank controls. No model training or robot subtraction. Query episodes
stay excluded from retrieval. Absent/uncertain assistant anchors clear to
unknown, not whole-window negative labels. A hash-bound `review.json` must
approve the generated seed revision before propagation.

Completed result: `annotations/seed_admission_pilot_20260918`.
Report: `docs/seed_admission_assisted_pilot_report_20260918.md`.
Read-only result verification (writes a validation sidecar):

```bash
python tools/object_annotation/validate_seed_admission.py
python -m unittest discover -s tools/object_annotation -p 'test_*.py'
```

For new runs, phases are `prepare --prompts PATH`, `seeds`, `run`, `finish`,
each with `--output NEW_ROOT`. Optional `--seed-revision` selects a separately
reviewed set, and `run --shard N --shards K` assigns disjoint cases. The current
script deliberately fixes the bounded case/candidate policy; it is not a
full-corpus entry point. The seed phase generates proposals, not accepted GT.
Do not mechanically write an approval sidecar without actually viewing them.
The final report separates near-seed development scores from longer unlabelled
visual checks; neither demonstrates whole-corpus semantic acceptance.
