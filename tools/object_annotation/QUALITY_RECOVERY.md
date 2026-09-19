# Manual mask audit and automatic tracking recovery

Scope: RGB object masks only, current pilot plus two existing pilot episodes.
No temporal grounding, subgoal logic, reconstruction or training. Inputs and
previous outputs are immutable. Use the `astribot-annotation` environment and
[existing official model installation](README.md).

## Human audit

For SSH-only users, download the self-contained 160 MiB local package; it runs
with Python 3.9+ standard library and a local browser, without GPU/models/pip.
See [local download/start/return instructions](../../docs/mask_audit_local_handoff_2026-09-16.md).
Package: `deliverables/astribot_mask_audit_420_20260916.zip`. Run `python3 start.py`
after extracting on your computer, then `python3 export_labels.py` to return
only labels. The server editor also remains accessible through SSH forwarding.

Frozen set: `annotations/mask_human_audit_420_20260916/manifest.json`.
It contains 420 losslessly exported RGB images, 140 per camera, in 21 clips of
20 sampled frames each (stride 5). Episodes: 002478/task24, 001608/task1,
002577/task24 (opposite side). There are 1,860 object/frame labels to review.
Frames were chosen before recovery results; this is a deliberately difficult
diagnostic set, not an independent or population-random test set. Condition tags
are only verified when a human labels them; clip selection alone does not prove
coverage of every requested condition.

Server (already running in tmux `mask-audit-ui-20260916`):

```bash
cd <SOURCE_WORKSPACE>
conda activate astribot-annotation
python tools/object_annotation/audit_server.py \
  annotations/mask_human_audit_420_20260916 --port 8765
```

Use VS Code Remote SSH Ports to forward server port 8765, or on your Mac with
HKU VPN connected:

```bash
ssh -N -L 8765:127.0.0.1:8765 yhwang@147.8.117.242
```

Then open <http://127.0.0.1:8765> in your local browser. The annotation server
listens only on server loopback; it is not a public website. Do not start a
second server on the same port if the existing one is running.

For every frame/object:

1. Draw only visible object pixels using polygon/brush/eraser. Exclude visible
   gripper surfaces; do not invent hidden/amodal object geometry. A physical
   overlap in projection does not mean the frontmost pixels have two visible
   labels. Boundary uncertainty can be noted or marked unknown.
2. Set `visible`, `partial_occlusion`, `fully_occluded`, `out_of_view`, or
   `unknown`. The last three require an empty visible-surface mask; visible
   and partial require a nonempty mask.
3. Confirm the object's stable physical instance ID across the clip/episode,
   and mark the applicable conditions. Do not relabel a watermelon as peach
   because the model did so. The current UI supports one instance per listed
   class; expand the schema before auditing multiple same-class instances.
4. Enter the annotator name, explicitly confirm manual review, and save. Save
   before moving to the next object/frame. Model predictions are not prefilled.

Current labels are atomic JSON files in `human_labels/`; all revisions remain in
`human_label_revisions.jsonl`. Do not programmatically populate these with SAM
predictions. An AI visual opinion is not independent human GT.

## Recovery behavior and comparison

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python tools/object_annotation/recovery_pilot.py \
  --output annotations/NEW_RECOVERY_PILOT
python tools/object_annotation/validate_recovery.py annotations/NEW_RECOVERY_PILOT
```

Recheck GPU availability first. A new output root is required; this bounded
runner does not implement partial resume. Defaults run exactly three episodes,
three cameras, stride 5. The pipeline uses fresh official GroundingDINO calls;
all candidate outputs and image hashes are recorded. SAM2 uses official
base-plus, BF16, mask logit > 0; DINO uses the existing FP32 span-scoring adapter.

The control and recovery arms get identical initial automatic seeds, confirmed
on two detector calls; both then advance causally on the same sampled images.
Control never re-seeds. Recovery may reset only its affected object's SAM2 state.
This is a controlled A/B, separate from the old **reviewed-seed bidirectional**
pilot. Do not attribute differences against that old run solely to recovery.

Triggers: low adjacent mask IoU, area/center jumps, empty masks, tiny border
residue, object-object overlap, or robot overlap warning. Active tracks also
receive a detector check every 15 sampled frames (~2.5 s); lost/suspicious
tracks can check every 3 sampled frames (~0.5 s). Defaults live in `recovery.py`.
At score >=0.40, two temporally consistent detections are required; same-class
ambiguity and cross-class box collisions cause abstention. These conservative
gates can also miss true objects; they are not tuned using the audit GT.

After at least three unsupported detector checks, sufficiently long empty
masks, tiny border residues, or low-presence severely unstable masks can end
the track. Status remains `lost`, with continued detection for re-entry.
Termination means **model-inferred loss**, not GT proof of physical out-of-view.
The next confirmed candidate starts a new segment with the same semantic track
ID. Robot overlap alone never removes object pixels or terminates a track.
Robot masks are unchanged from the previous pilot for episode002478; the two
extra episodes explicitly store null/unavailable robot masks.

JSONL retains RLE, derived bbox, scores, visible, layer, stable semantic track ID,
segment ID, status, seed provenance and suspicious flags. `*.events.jsonl`
records detector candidates and reasons. Suspicious flags on recovery rows
refer to the prediction **before** that frame's recovery decision. The validator
also computes diagnostics on final output masks. An initialized lost track has
an empty mask and predicted `visible=false`; a never-initialized object remains
`visible=null`. Neither value is a GT label.

## True quality metrics

After human labels have been saved, rerun into a new report directory:

```bash
python tools/object_annotation/evaluate_mask_audit.py \
  --audit annotations/mask_human_audit_420_20260916 \
  --run annotations/tracking_recovery_pilot_20260916 \
  --output annotations/mask_quality_metrics_REVIEW_VERSION
```

The evaluator checks label provenance/image hashes and outputs JSON, CSV and
Markdown for both arms, grouped by camera and manually tagged conditions:

- IoU/Dice: mean over visible/partially occluded object-frames; a missed object
  scores zero. Invisible GT is excluded from overlap averages, not assigned 1.
- Visibility accuracy: unknown GT excluded; unknown predictions count as wrong
  and are separately counted as abstentions.
- ID switches: class-agnostic Hungarian mask IoU >=0.1 assignment on fully
  labeled frames, then changes in predicted stable track ID per GT instance.
  Re-seeding a segment alone is not an ID switch. Semantic identity accuracy
  additionally catches consistently wrong class assignments.
- False-positive persistence: consecutive audited samples with a nonempty
  prediction on invisible GT, with span seconds and explicit lower-bound
  duration. No interpolation over gaps between audit clips.
- Out-of-view residue: positive-mask rate and pixel count when GT says
  `out_of_view`.
- Re-entry: GT out-of-view -> visible, with intended-object IoU >=0.5 within
  three sampled frames. Right-censored/re-occluded windows are excluded, and
  the eligible-event denominator is reported. Count of automatic re-seed
  events is **not** this success rate.

With zero human labels, overlap/accuracy/ID/re-entry scores are null/N/A and
status is `PENDING_HUMAN_GT`. Partial annotation is explicitly marked partial.
No camera/condition receives automatic pseudo-GT acceptance from model-only
consistency diagnostics. Review pixel errors and identity/visibility failures
before accepting any subset for downstream supervision.

Tests:

```bash
python tools/object_annotation/test_recovery.py
python tools/object_annotation/test_mask_audit.py
# Optional real-browser regression, synthetic temporary fixture only:
python -m pip install playwright==1.55.0
PLAYWRIGHT_BROWSERS_PATH=<SOURCE_WORKSPACE>/.cache/playwright python -m playwright install chromium
python tools/object_annotation/test_audit_browser.py
```
