# SAM2.1 object + robot mask pilot

This extends the bbox pilot without changing any existing annotations or source
RGB/HDF5/LeRobot data. Input scope is the existing task24 / episode_002478, three
cameras, 250 sampled frames per camera (stride 5). No training, temporal
grounding, subgoals, active-object logic or reconstruction is implemented.

## Run and inspect

For a fresh environment, follow [README installation and SAM2 setup](README.md).
The pinned requirements already include `pycocotools==2.0.8` for compressed RLE.

Use the existing `astribot-annotation` conda environment and a currently free GPU:

```bash
cd <SOURCE_WORKSPACE>
conda activate astribot-annotation
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python tools/object_annotation/mask_pilot.py \
  --source-run annotations/grounded_sam2_pilot_20260916 \
  --robot-prompts tools/object_annotation/config/robot_mask_prompts_pilot.json \
  --output annotations/NEW_MASK_RUN
python tools/object_annotation/audit_masks.py annotations/NEW_MASK_RUN
python tools/object_annotation/mask_qa.py annotations/NEW_MASK_RUN
python tools/object_annotation/test_masks.py
```

Use a new output directory; current pilot does not implement partial resume.
Frames reference the previous run's derived JPEG cache; never overwrite/delete
that cache while an annotation depends on it. Per-camera cache hashes and original
video frame/timestamp checks are included in acceptance. The checkpoint remains
official `sam2.1_hiera_base_plus.pt`, BF16 autocast, logit threshold 0.

Object initialization uses exactly the prior reviewed GroundingDINO boxes.
Robot GroundingDINO proposals were tested separately and produced false positives
on utensils/pot handles. `robot_mask_prompts_pilot.json` instead contains explicit
RGB-reviewed boxes/positive/negative points, with assisted provenance. This is
not fully automatic robot initialization. No physical/anatomical left/right
identity is asserted from screen position. Camera-local robot components are
retained and their union is the separate `robot_mask` layer.

## Files and schema

```text
task_24/episode_002478/
  head.objects.jsonl
  head.robot.jsonl
  head.robot_components.jsonl
  head.complete.json
  ... same for left_wrist and right_wrist
qa_masks/{head,left_wrist,right_wrist}.mp4
qa_robot_masks/{head,left_wrist,right_wrist}.mp4
qa_index.html
review_frames/{camera}/{frame_idx}.jpg
mask_quality.csv
object_diagnostics.jsonl
object_pair_overlaps.jsonl
robot_diagnostics.jsonl
acceptance.json
```

Each object/frame has a row with `episode_id`, `task_id`, `frame_idx`, original
camera `timestamp`, `camera`, `object_id`, `class_name`, `bbox_xyxy`, `mask`,
`confidence`, `visible`, image dimensions, `mask_area`, and provenance.
`bbox_xyxy` uses exclusive x1/y1 and is computed directly from positive pixels,
not copied from the seed. Masks are compressed COCO RLE:

```json
{"format":"coco_rle","size":[720,1280],"counts":"compressed ASCII counts"}
```

Counts use COCO's column-major encoding, not a flat row-major string. Example:

```python
import json
from pycocotools import mask as mask_utils

row = json.loads(open('head.objects.jsonl').readline())
if row['mask'] is not None:
    rle = row['mask']
    binary = mask_utils.decode({'size': rle['size'], 'counts': rle['counts'].encode('ascii')})
    assert binary.shape == (row['image_height'], row['image_width'])
```

Dense rows avoid confusing missing inference with invisibility:

- `visible=true`: model produced a nonempty mask, **not verified real visibility**.
- `visible=false`, `mask_status=predicted_empty`: decoded zero-area RLE, null bbox.
- `visible=null`, `mask_status=not_initialized`: mask/bbox/confidence are null.
  The prior unseeded left-wrist avocado is preserved this way for all 250 frames.

Object confidence = seed DINO score × SAM2 presence; robot component confidence =
SAM2 presence; robot union confidence = minimum presence among nonempty
contributors, or 0 when all empty. These are uncalibrated heuristics, **not**
mask IoU, segmentation certainty or reliability. High presence can accompany a
wrong mask. Components are anonymous, camera-local parts; their IDs are strings
and separate from numeric task object IDs. No robot mask is merged into object
rows and no robot subtraction/mutual exclusion/morphological cleanup is applied.

## QA and diagnostics

The HTML index links three combined videos and three robot-only videos. Original
RGB is left, predicted masks are right. Robot is cyan, objects have separate
colors, and object/robot overlap is magenta. Labels show class and confidence.
Videos run at 6 fps (30/stride); they are sampled-frame pilot results, not proof
of dense 30 Hz stability. Stills preserve specific failure frames.

Diagnostics include adjacent sampled mask IoU, mask area changes, inferred
visibility transitions, object/robot shared pixels and per-object overlap
fraction, and pairwise object-mask overlaps. They flag; they never delete masks.
Robot overlap is not ground-truth contamination: either prediction may be wrong.
Object-pair overlap is not by itself proof of identity switching. Inspect RGB
and the masks before interpreting either metric.

Acceptance verifies all RLE shape/areas, exact mask-derived boxes, null/empty
semantics, robot union, source timestamps, cache/output hashes and task whitelist.
The pilot object bbox output reproduces the previous bbox-only run exactly.
Engineering acceptance does **not** grant GT quality acceptance. See
`docs/object_robot_mask_pilot_report_2026-09-16.md` for findings.
