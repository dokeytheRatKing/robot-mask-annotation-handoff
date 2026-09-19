# Astribot object bounding-box annotation

Stage 1 only: per-task object detection on three independent RGB cameras.
No training, temporal grounding, subgoals, pose, reconstruction or depth use.
Raw HDF5, videos, Parquet and existing metadata are read-only inputs.

## Installation on <SOURCE_HOST>

Verified on driver 550.54.15, A100 80GB, CUDA toolkit 12.4, GCC 11.4,
Python 3.10, Torch 2.5.1+cu124 and torchvision 0.20.1+cu124. Environment:
`<ANNOTATION_ENV>`.
Existing training environments are not modified.

```bash
cd <SOURCE_WORKSPACE>
conda create -y -n astribot-annotation python=3.10 pip
conda activate astribot-annotation
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r tools/object_annotation/requirements.txt
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
git -C third_party/GroundingDINO checkout 856dde20aee659246248e20734ef9ba5214f5e44
CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 \
  python -m pip install --no-build-isolation --no-deps -e third_party/GroundingDINO
mkdir -p models/groundingdino
curl -fL --retry 3 -o models/groundingdino/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
python -c "from huggingface_hub import snapshot_download; snapshot_download('google-bert/bert-base-uncased',revision='86b5e0934494bd15c9632b12f734a8a67f723594',local_dir='models/groundingdino/bert-base-uncased',allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors'])"
python -m pip check
python tools/object_annotation/test_contract.py
```

Swin-T checkpoint SHA256:
`3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`.
Official repositories were cloned directly; CUDA extensions built successfully.
The inference adapter disables activation checkpointing (no gradients needed),
uses FP32 and local BERT assets. `HF_HUB_OFFLINE=1` works after downloads.

## Configurations

`config/objects.json` preserves all 24 object IDs and detailed English descriptions.
`config/task_objects.json` preserves the document's explicit task-to-object-ID column.
Object IDs are semantic object types, not automatically persistent instance IDs.
Multiple physical instances of the same type can yield multiple rows.

Known document ambiguities, **not silently fixed**:

- task23 describes a frying pan but omits object 5 from its object-ID column.
- task22 steps use a drying rack (8), which is absent from its explicit list.
- task26 lists kettle (1), but its written steps do not use one.
- task12/task23 list whole meat (17) and its two parts (22/23); these classes
  overlap visually and are not guaranteed distinguishable by open-vocabulary detection.
- task0 is defined but absent from the current dataset. It is not a valid pilot task.

The pilot tasks 1/9/24 do not depend on resolving these ambiguities. Clarify the
other tasks before expanding annotation to them. `config/prompt_overrides_visual_v2.json`
is a separate measured prompt experiment, not an accepted universal replacement.

Detailed descriptions are joined with periods **only for the selected task**.
Class assignment uses explicit noun-phrase character/token spans and the official
`create_positive_map_from_span` mean-token similarity. This avoids fuzzy label
matching to assign IDs. One winning class per query, class-wise NMS IoU=0.5.
`text_threshold` is not used in this official phrase-span mode; confidence is
not a calibrated probability. Baseline threshold=0.30, candidate floor=0.15.

## Input adapters

1. LeRobot root: use `meta/source_manifest.json` and `meta/info.json`, resolve
   the actual chunk/video paths, preserve canonical `episode_index`, map
   `head/left/right` to `head/left_wrist/right_wrist`. Task IDs come from
   `source_task_number`, **not** compact `task_index` or episode order.
2. Image folders: discover `task_N/episode_ID/{head,left_wrist,right_wrist}`
   anywhere under input; natural-sort JPEG/PNG names. Missing timestamps stay null.
3. Arbitrary layout: provide a JSON manifest with an `episodes` list:

```json
{"episodes":[{"episode_id":"episode_000","task_id":1,"fps":30,
 "cameras":{
  "head":{"video":"/absolute/path/head.mp4","timestamps":[123.0,123.03]},
  "left_wrist":{"images":["/absolute/left0.png","/absolute/left1.png"],"timestamps":[123.01,123.04]},
  "right_wrist":{"video":"/absolute/path/right.mp4","timestamps":[123.02,123.05]}
 }}]}
```

Supply one timestamp per actual frame; do not invent a shared camera clock.
For LeRobot, timestamps are original per-camera relative times from Parquet
plus `source_timestamp_start`. Video frame indices are canonical LeRobot rows.
No cross-camera pixel correspondence or depth alignment is performed here.

## Batch/pilot invocation

Recheck GPU ownership first. A new output directory is required.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python tools/object_annotation/annotate.py \
  --input datasets/lerobot/astribot_full_v21_rgb_h264 \
  --output annotations/my_new_pilot \
  --tasks 1 9 24 --episodes-per-task 5 --seed 20260916 \
  --stride 5 --threshold 0.30 --qa-per-task 2
python tools/object_annotation/summarize.py annotations/my_new_pilot
```

`--dry-run` scans/selects without loading models or writing annotation data.
`--stride 1` detects every frame; `--max-frames 90` limits a diagnostic probe.
`--episodes-per-task 0` explicitly selects all episodes of requested tasks;
full-corpus runs have **not** been started. `--prompt-overrides FILE` records
separate prompt descriptions and hashes. No directory layout changes needed.

`--resume` skips completed camera streams only if config/code/input-manifest/
checkpoint hashes match. Interrupted camera files have `.partial` suffixes;
they are restarted, never merged into completed results. To change prompts,
threshold or code, use a new output root. Completed runs snapshot their code,
resolved object descriptions, selection, package versions and provenance hashes.

## Output contract

```text
annotations/run/
  run_config.json, environment.json, selected_episodes.json
  task_01/episode_001608/
    head.jsonl, left_wrist.jsonl, right_wrist.jsonl
    head.frames.jsonl, head.candidates.jsonl, head.diagnostics.jsonl
    head.complete.json ...
  qa/task_01/episode_.../head.mp4 ...
  summary.json, measured_report.md, object_camera_threshold_metrics.csv
```

Every detection has `episode_id`, integer `task_id`, zero-based `frame_idx`,
original `timestamp` in seconds or null, `camera`, `object_id`, `class_name`,
pixel `bbox_xyxy`, `confidence`, `image_width`, `image_height`.
Coordinates are float, clipped to `[0,width] × [0,height]`, x1>x0/y1>y0;
right/bottom are continuous/exclusive image boundaries.
Absent objects generate **no fake box**. `*.frames.jsonl` records every
processed frame, including zero detections and absent IDs, so absence is
distinguishable from unprocessed data. Candidates are explicitly below-threshold
diagnostic artifacts and must not be mistaken for accepted annotations.

QA videos show original RGB beside bbox/class/confidence. Two episodes per task
are chosen by a recorded random seed; all three cameras are rendered. Sampled
QA runs at source fps/stride, retaining approximate episode duration.

## Temporal diagnostics

Adjacent **processed** frames use highest-confidence box per semantic object
class (not an instance tracker). Records include actual frame gap, IoU, center
distance divided by image diagonal, and max(area ratio, inverse area ratio).
Flag `IoU < .05`, center jump `> .15` diagonal, area ratio `> 3`, disappearance
or reappearance. Suspicious flags never delete or alter detection rows.
Multiple instances can cause highest-score switching; real movement, occlusion
and wrist motion can trigger flags. Coverage is **not recall**, and smooth
incorrect boxes can have excellent consistency. A separate stride-1 probe
checks actual consecutive frames; sampled pilot metrics must not be relabeled.

## Optional second stage (triggered by observed DINO failures)

Official Grounded-SAM-2 workflow reference:
https://github.com/IDEA-Research/Grounded-SAM-2
Official SAM2 implementation:
https://github.com/facebookresearch/sam2

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/Grounded-SAM-2
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
python -m pip install hydra-core==1.3.2 iopath==0.1.10
CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 SAM2_BUILD_ALLOW_ERRORS=0 \
  python -m pip install --no-build-isolation --no-deps -e third_party/sam2
mkdir -p models/sam2
curl -fL --retry 3 -o models/sam2/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
python tools/object_annotation/propose_seeds.py annotations/my_new_pilot episode_ID annotations/seed_review
# Inspect proposals, select genuine detector boxes in a reviewed manifest.
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
python tools/object_annotation/sam2_refine.py \
  --dino-run annotations/my_new_pilot --seeds annotations/seed_review/reviewed_seeds.json \
  --output annotations/my_sam2_pilot --stride 5
```

Review proposals before using them: highest-score detections can be wrong.
The implemented fallback accepts verified DINO seeds, propagates independently
forward/backward with official SAM2.1, and exports tight positive-mask boxes in
the same required JSONL schema. Empty masks create no rows. It adds backend,
seed frame and score provenance; confidence is `seed DINO score × SAM2 presence`,
an **uncalibrated heuristic**, not a new DINO measurement. The fallback is
assisted initialization, not validated fully automatic seed selection.
Unseeded objects remain absent. SAM2 can still drift through occlusion.

Derived JPEG working caches live only in the annotation output root. They are
not model-training data and do not replace original RGB. The bbox-stage command
does not export masks/poses/depth supervision. Original videos remain untouched.

## Segmentation mask pilot extension

The subsequent user-requested mask pilot retains SAM2.1 object masks and exports
separate robot masks, compressed COCO RLE, derived bboxes and combined/robot-only
QA videos. See [MASKS.md](MASKS.md) for installation, commands, schema and limits.
Output: `annotations/grounded_sam2_mask_pilot_20260916/qa_index.html`.
Engineering acceptance passed; visual quality is not yet sufficient for direct
GT use. Robot seeds are assisted and anatomical left/right identity is unverified.
Findings: `docs/object_robot_mask_pilot_report_2026-09-16.md`.

The follow-up [quality audit and tracking recovery](QUALITY_RECOVERY.md) adds a
420-frame manual mask editor, human-GT evaluator and bounded causal A/B recovery
experiment. Real IoU/Dice/identity/re-entry metrics remain pending until people
complete the audit labels; automatic consistency is not a substitute for GT.
