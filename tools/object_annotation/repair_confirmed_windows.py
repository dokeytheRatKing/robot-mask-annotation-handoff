"""Bounded bidirectional SAM2 repair from confirmed masks; never overwrite drafts."""
import argparse
from collections import defaultdict
import gc
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

from annotate import PROJECT, QAVideo, sha, write_json
from evaluate_confirmed_batch import PART_IDS, prediction_mask
from masks import bbox, decode, record


def painted(image, rows):
    result = image.copy()
    for row in rows:
        if row['mask'] is None or not row['visible']:
            continue
        binary = decode(row['mask'])
        color = np.random.default_rng(row['object_id'] + 7).integers(65, 245, 3)
        result[binary] = (result[binary] * .52 + color * .48).astype(np.uint8)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, tuple(map(int, color)), 1)
        x0, y0, x1, y1 = row['bbox_xyxy']
        cv2.rectangle(result, (x0, y0), (x1, y1), tuple(map(int, color)), 1)
    return result


def comparison(image, old, new, caption):
    canvas = np.zeros((392, 1280, 3), np.uint8)
    for i, panel in enumerate((image, painted(image, old), painted(image, new))):
        canvas[32:272, i*426:(i+1)*426] = cv2.resize(panel, (426, 240))
    cv2.putText(canvas, caption, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    cv2.putText(canvas, 'RGB                              OLD FROZEN                       REVIEWED-SEED SAM2 DRAFT',
                (8, 293), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
    for i, row in enumerate(new):
        color = tuple(map(int, np.random.default_rng(row['object_id'] + 7).integers(65, 245, 3)))
        score = row.get('confidence')
        suffix = f' p={score:.2f}' if score is not None else ' unknown' if row['mask'] is None else ' seed'
        cv2.putText(canvas, row['class_name']+suffix, (8+(i%4)*315, 318+(i//4)*23),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, color, 1)
    return canvas


def load_baseline(root, case, start, end):
    frames = defaultdict(lambda: defaultdict(list))
    for layer in ('objects', 'robot_parts'):
        path = root / case['source_files'][layer]
        assert sha(path) == case['source_hashes'][layer]
        with gzip.open(path, 'rt') as f:
            for line in f:
                row = json.loads(line)
                if start <= row['frame_idx'] <= end:
                    frames[row['frame_idx']][layer].append(row)
                if row['frame_idx'] > end:
                    break
    assert set(frames) == set(range(start, end+1))
    return frames


def suspicious(previous, binary, robot):
    flags = []
    if previous is not None and previous.any() and binary.any():
        union = int((previous | binary).sum())
        if (previous & binary).sum() / union < .1:
            flags.append('low_adjacent_mask_iou')
        ratio = binary.sum()/previous.sum()
        if ratio > 3 or ratio < 1/3:
            flags.append('area_jump')
        a, b = bbox(previous), bbox(binary)
        jump = np.linalg.norm(np.array(a[:2])+a[2:]-np.array(b[:2])-b[2:]) / 2
        if jump > .15 * np.hypot(*binary.shape):
            flags.append('center_jump')
    if previous is not None and bool(previous.any()) != bool(binary.any()):
        flags.append('predicted_presence_transition_unverified')
    if binary.any() and (binary & robot).sum()/binary.sum() > .2:
        flags.append('object_robot_overlap_not_subtracted')
    return flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accepted', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--production', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--radius', type=int, default=30)
    a = p.parse_args()
    assert a.radius > 0 and len(set(a.cases)) == len(a.cases)
    labels_path = a.accepted/'accepted_masks.jsonl'
    labels = defaultdict(list)
    for line in labels_path.read_text().splitlines():
        row = json.loads(line)
        if row['sample_id'] in a.cases:
            assert row['human_confirmed']
            labels[row['sample_id']].append(row)
    assert set(labels) == set(a.cases)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(20260918)
    checkpoint = PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt'
    model_config = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    predictor = build_sam2_video_predictor(model_config, str(checkpoint), device='cuda',
        apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0', '++model.non_overlap_masks=false'])
    config = dict(cases=a.cases, radius=a.radius, checkpoint=str(checkpoint),
                  checkpoint_sha256=sha(checkpoint), model_config=model_config,
                  accepted_sha256=sha(labels_path), script_sha256=sha(__file__),
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  scoring='No nonseed GT: no repaired IoU claim. Seed frames excluded from quality evaluation.',
                  policy='No robot subtraction. Seed-absent objects are UNKNOWN at nonseed frames.',
                  direction='Independent forward/reverse states; dense original frames; offline assisted annotation.')
    write_json(a.output/'run_config.json', config)
    summaries = []
    for sid in a.cases:
        started = time.perf_counter()
        case_path = a.snapshot/sid/'case.json'
        case = json.loads(case_path.read_text())
        reference = cv2.imread(str(a.snapshot/sid/'rgb.png'))
        assert sha(a.snapshot/sid/'rgb.png') == labels[sid][0]['provenance']['source_image_sha256']
        cap = cv2.VideoCapture(case['source_video'])
        total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        seed = case['frame_idx']; start, end = max(0, seed-a.radius), min(total-1, seed+a.radius)
        cache = a.output/sid/'frames'; cache.mkdir(parents=True)
        images = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            for idx in range(start, end+1):
                ok, image = cap.read(); assert ok
                if idx == seed:
                    assert np.array_equal(image, reference), 'Seed video/image alignment failed'
                images.append(image)
                assert cv2.imwrite(str(cache/f'{idx-start:05d}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 98])
        finally:
            cap.release()
        baseline = load_baseline(a.production, case, start, end)
        editable = [r for r in labels[sid] if r['group'] != 'robot']
        positive = {r['object_id']: decode(r['mask']) for r in editable
                    if r['visible'] and not r['exclude_from_metrics']}
        assert positive
        prediction, confidence = {}, {}
        torch.cuda.reset_peak_memory_stats()
        inference_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for reverse in (False, True):
                state = predictor.init_state(str(cache), offload_video_to_cpu=True, offload_state_to_cpu=True)
                for oid, binary in sorted(positive.items()):
                    predictor.add_new_mask(state, seed-start, oid, binary)
                for local, oids, logits in predictor.propagate_in_video(state, start_frame_idx=seed-start, reverse=reverse):
                    if reverse and local >= seed-start:
                        continue
                    prediction[local] = dict(zip(oids, (logits[:, 0] > 0).cpu().numpy()))
                    for obj_idx, oid in enumerate(oids):
                        outputs = state['output_dict_per_obj'][obj_idx]
                        cached = outputs['cond_frame_outputs'].get(local)
                        if cached is None:
                            cached = outputs['non_cond_frame_outputs'][local]
                        confidence[local, oid] = float(cached['object_score_logits'].float().sigmoid().reshape(-1)[0])
                del state
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-inference_start
        assert set(prediction) == set(range(end-start+1))
        writers = {g: QAVideo(a.output/'qa'/f'{sid}_{g}.mp4', fps) for g in ('objects', 'robot_parts')}
        sheets = defaultdict(list); previous = {}; flag_counts = defaultdict(int)
        path = a.output/sid/'predictions.jsonl'
        contacts = set(np.linspace(start, end, 5, dtype=int).tolist()) | {seed}
        with path.open('x') as out:
            for local, image in enumerate(images):
                idx = local+start
                source = baseline[idx]['objects'][0]
                timestamp = source['timestamp']
                assert all(r['timestamp'] == timestamp for rows in baseline[idx].values() for r in rows)
                parts = [prediction[local][oid] for oid in PART_IDS if oid in positive]
                union = np.logical_or.reduce(parts) if parts else np.zeros(image.shape[:2], bool)
                if idx == seed:
                    union = decode(next(r for r in labels[sid] if r['group'] == 'robot')['mask'])
                rows, old = [], []
                for label in editable:
                    oid = label['object_id']
                    common = {k: label[k] for k in ('episode_id', 'task_id', 'camera', 'object_id',
                               'class_name', 'group', 'merged_object_ids', 'image_width', 'image_height')}
                    common.update(frame_idx=idx, timestamp=timestamp, seed_frame=seed,
                                  confidence=None, human_confirmed=idx == seed,
                                  provenance=dict(mode='confirmed_seed' if idx == seed else 'reviewed_seed_sam2_draft',
                                      seed_case=sid, accepted_sha256=config['accepted_sha256'],
                                      source_video=case['source_video'], config='../run_config.json'))
                    flags = []
                    if idx == seed:
                        binary = decode(label['mask'])
                        row = dict(common, **record(binary), visibility=label['visibility'])
                    elif oid in positive:
                        binary = prediction[local][oid]
                        flags = suspicious(previous.get(oid), binary,
                                           union if label['group'] == 'objects' else np.zeros_like(union))
                        row = dict(common, **record(binary), visibility='predicted_visible' if binary.any() else 'unknown')
                        row['confidence'] = confidence[local, oid]
                    else:
                        binary = None
                        flags = ['no_positive_seed_visibility_unknown']
                        row = dict(common, mask=None, bbox_xyxy=None, mask_area=None, visible=None,
                                   visibility='unknown', mask_status='unseeded_unknown')
                    previous[oid] = binary
                    row.update(suspicious_flags=flags,
                               confidence_semantics='SAM2 presence only; not calibrated quality; null for seed/unknown')
                    for flag in flags:
                        flag_counts[flag] += 1
                    out.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                    old.append(dict(common, **record(prediction_mask(label, baseline[idx]))))
                robot_label = next(r for r in labels[sid] if r['group'] == 'robot')
                robot = {k: robot_label[k] for k in ('episode_id', 'task_id', 'camera', 'object_id', 'class_name', 'group')}
                robot.update(frame_idx=idx, timestamp=timestamp, **record(union), confidence=None,
                             visibility='predicted_visible' if union.any() else 'unknown',
                             human_confirmed=idx == seed, derived_from=list(PART_IDS),
                             provenance=dict(mode='four_named_parts_union', seed_case=sid),
                             suspicious_flags=[] if idx == seed else ['unseeded_parts_may_enter'])
                out.write(json.dumps(robot, allow_nan=False)+'\n')
                for group, writer in writers.items():
                    canvas = comparison(image, [r for r in old if r['group'] == group],
                        [r for r in rows if r['group'] == group],
                        f'{sid} frame={idx} {group} | '+('SEED (NOT EVALUATION)' if idx == seed else 'NONSEED, UNCONFIRMED'))
                    writer.add(canvas)
                    if idx in contacts:
                        sheets[group].append(canvas)
            for writer in writers.values():
                writer.close()
        for group, panels in sheets.items():
            assert cv2.imwrite(str(a.output/'qa'/f'{sid}_{group}.jpg'), cv2.vconcat(panels))
        summary = dict(case_id=sid, start=start, end=end, seed=seed, frames=len(images), fps=fps,
                       nonseed_frames=len(images)-1, positive_seed_ids=sorted(positive),
                       unseeded_ids=[r['object_id'] for r in editable if r['object_id'] not in positive],
                       inference_seconds=inference_seconds, inference_fps=len(images)/inference_seconds,
                       total_seconds=time.perf_counter()-started,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       flags=dict(flag_counts), output_sha256=sha(path), source_case_sha256=sha(case_path))
        summaries.append(summary); write_json(a.output/'progress.json', summaries)
        print(json.dumps(summary), flush=True)
        del prediction, images; gc.collect(); torch.cuda.empty_cache()
    write_json(a.output/'run_complete.json', dict(status='COMPLETE', cases=summaries,
               frames=sum(s['frames'] for s in summaries), nonseed_frames=sum(s['nonseed_frames'] for s in summaries),
               numerical_repair_quality=None, reason='No independent nonseed GT in these windows'))


if __name__ == '__main__':
    main()
