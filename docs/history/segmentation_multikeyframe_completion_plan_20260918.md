# Full-corpus multi-keyframe segmentation completion plan

Date: 2026-09-18

## Target and constraints

Produce more accurate segmentation for all 5,803 canonical episodes, all three
cameras and all frames. Detect task-relevant objects only, plus independent
left/right arm and EE layers. Use automatic or assistant-assisted multi-keyframe
reseeding and SAM2; no training, reconstruction or temporal grounding.
GPU1 is excluded. Allowed devices: 0,2,3,4,5,6,7,8,9, after occupancy checks.

The user's target is to keep images requiring precise human annotation within
100 wherever possible. Do NOT interpret this as permission for another100
manual images. The existing420-frame and87-frame accepted returns are reusable
references, not new assignments. Their review counts do not establish the
number actually hand-drawn: the earlier78 manual entries are labels, not
necessarily distinct images. Historical manual-frame accounting remains to
be reconciled before requesting another batch. Start with zero new requests.

Count precise human work by unique `(episode_id, camera, frame_idx)`, regardless
of objects, edits or repeat exports. Separately record assistant/SAM proposals,
human verification, human pixel corrections and unresolved cases. Do not hide
manual corrections inside a verification count. No budget expansion without
the user's decision; retain unknowns if a reliable answer cannot be obtained.

## Current evidence

The frozen full automatic draft is structurally complete, not semantically
accepted. The87 deliberately difficult reviewed frames reveal substantial
misses and named head robot-part gaps. Single-seed propagation repaired several
local problems but also changed basket identity to avocado. Multiple inspected
keyframes repaired that observed failure. Evidence currently covers short
windows, not long episodes or general re-entry recovery.

Existing building blocks: atomic per-stream jobs and versioned full-run assets
in `tools/object_annotation/full_segmentation.py`; bounded reviewed propagation
in `repair_confirmed_windows.py`; assistant-guided seeds and bidirectional
propagation in `reseed_window_objects.py`; structural validators and QA exports.
The new multi-keyframe experiment is NOT yet integrated into the corpus runner.

## Recovery loop

1. Reuse immutable accepted masks, task object lists and identity exemplars.
   Never transplant a mask's coordinates to another camera or episode.
2. Propose seeds at clear views, reappearance, disagreement and segment
   boundaries. Use GroundingDINO plus image SAM2; check object identity as well
   as mask shape. Detector/SAM confidence alone is not acceptance evidence.
3. Propagate from multiple seeds in bounded overlapping temporal windows.
   Compare forward/backward masks, maintain physical identity, and inspect
   boundary consistency. Offline future-frame use is explicit.
4. Trigger local repair on presence gaps, abrupt shape/position changes,
   competing object overlap or anatomical ambiguity. Monitor persistent misses
   separately: a consistently empty mask can appear temporally stable.
5. Retry locally with additional identity-checked seeds. If repeated automatic
   attempts disagree, stop retrying and enqueue context RGB, crops and overlays
   for assistant visual inspection. Deduplicate similar failure cases.
6. The assistant supplies points/boxes or corrected identity/visibility and
   inspects SAM2's resulting mask before propagation. Only genuinely unresolved
   identity, visibility or boundaries reach the user. Offer a proposal and
   context first; request precise drawing only when necessary.

Never infer whole-window absence from one negative seed. Keep unresolved
visibility explicit. Robot overlap is a warning, not subtraction. Left/right
identity must not follow image x-position. Unseen flange boundaries remain
unresolved rather than guessed. Preserve accepted seeds and all provenance.

## Execution order and release gate

First generalize the local recovery tool to longer clips, entering objects and
robot-part seeds. Test dense recovery over full selected episodes, covering
both wrist cameras, head, grasp contact and re-entry; use nonseed visual checks.
Then run a bounded multi-task canary before selective corpus repair on the nine
allowed GPUs. Do not spend a full run repeating a known-bad identity rule.

Keep a versioned base-plus-repair manifest: unchanged draft segments remain
traceable; accepted human seeds take precedence; promoted repairs reference
their source and seed hashes. Publish one resolved annotation view only after
checking all frame keys, task IDs, timestamps, mask geometry and layer unions.
Unknown is not an empty negative training label. Every frame must have a
traceable processing/quality status, including unresolved frames.

Quality assessment combines difficult-case regression and independently sampled
nonseed frames across tasks/cameras. Never score initialization seeds as recovery
success. Reusing a reference for tuning removes its status as independent test
evidence. Report actual labeled sample counts, IoU/Dice and visibility only where
reference labels exist; separately report visual review and unknown coverage.
Measure ID switches/re-entry success only with labeled temporal reference, not
from the algorithm's own reacquisition events. Do not claim a numerical global
accuracy guarantee from the present biased audit set.

Human effort reduction is a workflow target, not proof that every ambiguous
pixel can be resolved under100 images. If quality and that budget conflict,
report residual uncertainty and affected scope instead of silently accepting
bad masks or increasing human work.

## Operational limits

GPU workers can run detection/propagation and produce deduplicated review queues.
Assistant multimodal review currently occurs in active sessions; no unattended
GPT reviewing service is installed. Do not claim background GPT review exists
or add a paid remote inference dependency without a separate decision.
This document records the revised execution target; no new full-corpus job or
human annotation batch was launched by this planning update.
