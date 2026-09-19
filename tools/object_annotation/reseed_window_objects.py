"""Assistant-selected keyframes for bounded object re-seeding; robots unchanged."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from annotate import PROJECT, QAVideo, sha, write_json
from masks import decode, record
from repair_confirmed_windows import comparison

MODEL = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
WEIGHTS = PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['seeds', 'propagate'], required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()); sid = cfg['case_id']
    complete = json.loads((a.parent/'run_complete.json').read_text())
    case = next(c for c in complete['cases'] if c['case_id'] == sid)
    rows = defaultdict(dict)
    parent_path = a.parent/sid/'predictions.jsonl'
    assert sha(parent_path) == case['output_sha256']
    for line in parent_path.read_text().splitlines():
        row = json.loads(line); rows[row['frame_idx']][row['object_id']] = row
    cache = a.parent/sid/'frames'
    image_at = lambda idx: cv2.imread(str(cache/f'{idx-case["start"]:05d}.jpg'))
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(20260918)
    if a.phase == 'seeds':
        a.output.mkdir(parents=True, exist_ok=False)
        predictor = SAM2ImagePredictor(build_sam2(MODEL, str(WEIGHTS), device='cuda', apply_postprocessing=False))
        seeds = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for prompt in cfg['keyframes']:
                idx, oid = prompt['frame_idx'], prompt['object_id']
                assert case['start'] <= idx <= case['end'] and idx != case['seed']
                image = image_at(idx); reference = rows[idx][oid]
                score = None
                if prompt.get('absent'):
                    binary = np.zeros(image.shape[:2], bool)
                else:
                    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    masks, scores, _ = predictor.predict(box=np.array(prompt['box']),
                        point_coords=np.array(prompt['points']), point_labels=np.array(prompt['point_labels']),
                        multimask_output=False)
                    binary = masks[0].astype(bool); score = float(scores[0])
                    for component in prompt.get('components', []):
                        component_masks, _, _ = predictor.predict(box=np.array(component['box']),
                            point_coords=np.array(component['points']), point_labels=np.array(component['point_labels']),
                            multimask_output=False)
                        binary |= component_masks[0].astype(bool)
                    assert binary.any()
                row = dict(reference, **record(binary))
                row.update(confidence=None, human_confirmed=False, visibility='assistant_visible' if binary.any() else 'assistant_not_visible',
                           provenance=dict(mode='assistant_visual_prompt_sam2_image', prompt=prompt,
                                           config_sha256=sha(a.config), predicted_iou=score))
                seeds.append(row)
                panel = comparison(image, [reference], [row], f'{sid} frame={idx} object={oid} ASSISTANT SEED PROPOSAL')
                cv2.imwrite(str(a.output/f'seed_{idx}_{oid}.jpg'), panel)
        write_json(a.output/'seeds.json', seeds)
        write_json(a.output/'config.json', dict(config=cfg, config_sha256=sha(a.config),
            parent_sha256=sha(parent_path), model_sha256=sha(WEIGHTS), script_sha256=sha(__file__)))
        print('Seed proposals ready for assistant visual inspection', flush=True)
        return
    saved = json.loads((a.output/'config.json').read_text())
    assert saved['parent_sha256'] == sha(parent_path) and saved['config_sha256'] == sha(a.config)
    assert saved['script_sha256'] == sha(__file__)
    seed_rows = json.loads((a.output/'seeds.json').read_text())
    review = json.loads((a.output/'seed_visual_review.json').read_text())
    assert review['seeds_sha256'] == sha(a.output/'seeds.json') and review['decision'] == 'use_as_assisted_seeds'
    seeds = {(r['frame_idx'], r['object_id']): r for r in seed_rows}
    objects = sorted({r['object_id'] for r in seed_rows})
    for oid in objects:
        seeds[case['seed'], oid] = rows[case['seed']][oid]
    predictor = build_sam2_video_predictor(MODEL, str(WEIGHTS), device='cuda', apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0', '++model.non_overlap_masks=false'])
    directions = {}; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for reverse in (False, True):
            state = predictor.init_state(str(cache), offload_video_to_cpu=True, offload_state_to_cpu=True)
            for (idx, oid), row in sorted(seeds.items()):
                predictor.add_new_mask(state, idx-case['start'], oid, decode(row['mask']))
            output = {}
            for local, oids, logits in predictor.propagate_in_video(state,
                    start_frame_idx=case['end']-case['start'] if reverse else 0, reverse=reverse):
                for obj_idx, (oid, binary) in enumerate(zip(oids, (logits[:, 0] > 0).cpu().numpy())):
                    cached = state['output_dict_per_obj'][obj_idx]
                    value = cached['cond_frame_outputs'].get(local)
                    if value is None:
                        value = cached['non_cond_frame_outputs'][local]
                    output[local+case['start'], oid] = (binary, float(value['object_score_logits'].float().sigmoid().reshape(-1)[0]))
            directions[reverse] = output
            del state
    torch.cuda.synchronize(); seconds = time.perf_counter()-started
    writer = QAVideo(a.output/'objects.mp4', case['fps']); panels = []
    target = a.output/'predictions.jsonl'
    with target.open('x') as out:
        for idx, original in sorted(rows.items()):
            new = []
            for oid, old in original.items():
                row = dict(old)
                if oid in objects and idx != case['seed']:
                    nearest = min((key for key in seeds if key[1] == oid), key=lambda key: (abs(key[0]-idx), key[0]))
                    if (idx, oid) in seeds:
                        row = dict(seeds[idx, oid])
                    else:
                        binary, confidence = directions[nearest[0] > idx][idx, oid]
                        row.update(**record(binary), confidence=confidence, human_confirmed=False,
                                   visibility='predicted_visible' if binary.any() else 'unknown',
                                   provenance=dict(mode='assistant_multiseed_sam2', nearest_seed=nearest[0],
                                       parent_run=str(a.parent), seeds_sha256=sha(a.output/'seeds.json')),
                                   suspicious_flags=['assisted_draft_requires_validation'])
                out.write(json.dumps(row, allow_nan=False)+'\n'); new.append(row)
            panel = comparison(image_at(idx), [r for r in original.values() if r['group'] == 'objects'],
                [r for r in new if r['group'] == 'objects'],
                f'{sid} frame={idx} | CENTER: SINGLE-SEED / RIGHT: MULTISEED (NOT GT)')
            writer.add(panel)
            if idx in (case['start'], case['start']+15, case['seed'], case['end']-15, case['end']):
                panels.append(panel)
    writer.close(); cv2.imwrite(str(a.output/'overview.jpg'), cv2.vconcat(panels))
    write_json(a.output/'run_complete.json', dict(status='COMPLETE', frames=len(rows), inference_seconds=seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, output_sha256=sha(target),
        unchanged_robot=True, human_confirmed_nonseed=False, quality_metrics=None,
        excluded_quality_frames=sorted({key[0] for key in seeds}), seed_manifest_sha256=sha(a.output/'seeds.json')))


if __name__ == '__main__':
    main()
