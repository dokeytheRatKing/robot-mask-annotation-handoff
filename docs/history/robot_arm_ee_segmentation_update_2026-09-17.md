# Robot Arm / EE Segmentation Update

## Contract

- Independent robot layer: `left_arm`, `right_arm`, `left_ee`, `right_ee`,
  plus `robot_unknown`. Left/right means physical robot side, not screen order.
- The flange itself belongs to **arm**. The tool side, including the gripper
  housing and both fingers, belongs to **EE**. EE is not only the black fingers.
- Object masks remain independent. No object-minus-robot subtraction.
- Unobserved or ambiguous anatomical boundaries must not become invented part
  labels. Robot-only union and raw anatomical candidates are retained separately.
- All automatic predictions remain unconfirmed drafts, not calibrated GT.

## Implementation

Code: `tools/object_annotation/full_segmentation.py`, `robot_parts.py`,
`identity_guard.py`; config: `config/robot_parts.json`.

Paired RGB and measured joints in task1 episodes 001603 and 001684 establish
the initial head-camera physical side mapping. Left/right joint total motion
is 13.24/0.185 and 0.553/9.47 respectively. Initial image-left is physical left
in these verified views. This does NOT authorize sorting crossed tracks by x.

SAM2.1 base-plus creates part seeds from inspected RGB prompts. The visible
mounting interface has an approximately 4px uncertainty band; it is not an
engineering-calibrated flange contour. Conservative foreground SIFT/RANSAC
registration transfers seeds near known initialization postures. Wrist-mounted
fingers have camera-side point prompts, with a frozen robot-appearance check.
Subsequent propagation retains IDs. No training or reconstruction is performed.

Crucially, a head-view initialization showing EE but no flange is insufficient
to label new forearm pixels later in the video. Such tracks become
`robot_unknown`, with their original side-labelled candidates retained. Within-
robot part overlaps also become unknown; task-object masks are never edited.

## Output

Versioned full-corpus root: `annotations/segmentation_full_robot_20260917`.
Canonical source: 5,803 episodes, 29 tasks, 17,409 camera streams and
9,357,261 camera-frames. Stride is 1; source videos and HDF5 are unchanged.

Each `task_XX/episode_xxxxxx/camera/attempt_NNN/` contains:

- `objects.jsonl.gz`: dense task-conditioned object predictions.
- `robot_parts.jsonl.gz`: five part rows per frame, original-resolution RLE,
  mask-derived bbox, timestamp, confidence, physical side, flags and provenance.
- `robot.jsonl.gz`: union of robot parts and unknown regions only.
- `robot_candidates.jsonl.gz`: unresolved part candidates, not accepted labels.
- `detections.jsonl.gz`, `events.jsonl.gz`, `review.jsonl.gz`: recovery evidence
  and deduplicated review requests. Review flags are not automatic deletion.
- `complete.json`: frame counts and SHA256 hashes. Completion means processing,
  not semantic acceptance. Final files replace `.partial` only after success.

Code, configs, references, appearance gallery, model revisions and checkpoint
hashes are frozen per run. Retried attempts preserve previous results.

## Quality Findings

The initial full draft was paused at 82 streams / 39,298 camera-frames after
visual review found robot-as-kettle and gripper-as-fruit false identities. It
remains under `annotations/segmentation_full_20260917` with `STOP`; it is not
accepted full annotation. No original data was changed.

Competitive robot/background grounding and an official frozen DINOv2 ViT-S/14
exemplar lookup now veto several observed identity errors. This is a heuristic,
not an independent accuracy measurement; strict vetoes can miss real objects.
Reviewed masks take priority over automatic seeding throughout their segment.
Reference masks are never presented as independent evaluation predictions.

Robot pilot r2 completed 7,455 frames / 12 streams and passed dense-frame,
timestamp, RLE and hash validation. It exposed incomplete robot registration.
Pilot r3 added EE templates and wrist prompts; visual review then caught EE
expansion onto newly visible forearms. The final r5 adds the anatomical guard
and a more specific mounted-finger check. It completed the same four episodes,
capped at 360 frames each: 4,059 frames / 12 streams, zero failed streams.
Dense frame/timestamp/RLE/bbox/hash validation passed. A separate exhaustive
check passed robot-union equality and part disjointness on all 4,059 frames.
All 34 annotation unit tests passed. The production code/assets exactly match
the final preflight snapshot; only the preflight episode manifest is bounded.

The first r1 attempt failed at a BF16-to-NumPy interface in the appearance
lookup. The lookup now computes similarities explicitly in FP32. Failed logs
are retained; those attempts are not counted as successful annotation. r4 also
exposed a capped-source timestamp-length assertion; original source length and
inference cap are now passed separately, verified by r5 and a regression test.

QA artifacts: `annotations/robot_parts_qa_20260917/`, including `report.json`,
48 original/overlay comparisons, two overview sheets and three MP4s for
episode001684. All three MP4s passed full FFmpeg decode. Downloadable preview:
`deliverables/astribot_robot_parts_preview_20260917.zip`.

On this bounded sample, all 1,353 frames of each wrist camera had a predicted
own-side EE mask. Head-view named parts were present in 513/1,353 frames; other
visible robot regions were retained as unknown. These are coverage counts,
NOT recall, correctness or IoU. Visual checks confirm the observed-flange
head example, mounted fingers and hidden-flange fallback work as intended.
Opposite-arm coverage in wrist views remains incomplete.

Peak Torch allocated memory was 3.062 GiB per worker. Summed processing time
was 632.37 worker-seconds for 4,059 frames, excluding model startup. This tiny,
two-task sample does not establish a full-corpus ETA.

No human robot-part masks exist, so robot IoU, Dice and anatomical accuracy
are **not measured**. Head examples with observed flange boundaries support
separate arm/EE candidates. Unseen boundaries, wrist opposite-arm coverage,
occlusion and apparent arm/EE drift remain review cases. Do not treat the
four-class output as uniformly reliable reconstruction supervision.

## Operations

Use `<ANNOTATION_ENV>/bin/python`.
All GPU 0-9 are user-authorized; unrelated resident processes are left alone.
The full queue was launched after r5 validation in detached tmux session
`segmentation-full-robot-20260917`, with one worker on each GPU0-9. The old
82-stream diagnostic remains stopped. Recheck live state before reporting
progress; launch does not imply completion or semantic acceptance.

```bash
cd <SOURCE_WORKSPACE>
<ANNOTATION_ENV>/bin/python tools/object_annotation/full_segmentation.py status --output annotations/segmentation_full_robot_20260917
```

`progress.json` and `logs/gpu0.log` through `gpu9.log` provide live state.
Create `STOP` in the run root to finish current streams without taking new
jobs. Before resuming, move that marker aside and restart the frozen supervisor.
Never edit a started run's frozen code/config. Review overrides are per-stream
snapshots; requeue only completed/failed streams, not active leases.

The review JSONL is a queue, not a background GPT vision service. Visual review
is performed during active assistant sessions; unresolved masks must not be
silently declared accepted when the worker queue finishes.

Official appearance model source: [DINOv2](https://github.com/facebookresearch/dinov2).
Local weights: `models/dinov2/dinov2_vits14_pretrain.pth`; no additional model
training, temporal grounding, depth processing or public web service was added.
