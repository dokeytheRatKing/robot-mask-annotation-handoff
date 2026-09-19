"""Episode-excluded detector-only vs seed-bank ranking on fixed reviewed frames."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from annotate import BASE, PROJECT, sha, write_json
from masks import decode, record
from seed_bank import PARTS, SeedBank, accepted_rows, crops, frame_key, rank_score, read_image


def metrics(pred, truth):
    inter = int((pred & truth).sum()); union = int((pred | truth).sum())
    return dict(iou=inter/union if union else 0,
                precision=inter/int(pred.sum()) if pred.any() else 0,
                recall=inter/int(truth.sum()) if truth.any() else 0)


def selected_queries():
    rows = accepted_rows()
    selected = [r for r in rows if r['source_set'] == 'accepted87']
    # Add fruit transitions from the other accepted set, not only its easy start.
    old = [r for r in rows if r['episode_id'] == 'episode_002577']
    for camera in ('head', 'left_wrist', 'right_wrist'):
        indices = sorted({r['frame_idx'] for r in old if r['camera'] == camera})
        chosen = {indices[i] for i in np.linspace(0, len(indices)-1, min(4, len(indices)), dtype=int)}
        selected.extend(r for r in old if r['camera'] == camera and r['frame_idx'] in chosen)
    groups = defaultdict(list)
    for r in selected:
        if not r.get('exclude_from_metrics'):
            groups[frame_key(r)].append(r)
    return dict(sorted(groups.items()))


def summarize(rows):
    result = {}
    for name, filtered in [('all', rows), ('objects', [r for r in rows if r['object_id'] < 1000]),
                           ('robot_parts', [r for r in rows if r['object_id'] >= 1100])]:
        result[name] = {}
        for mode in ('detector', 'rgb', 'masked', 'hybrid'):
            visible = [r for r in filtered if r['gt_visible']]
            covered = [r for r in visible if r['reference_available']]
            absent = [r for r in filtered if not r['gt_visible']]
            possible = [r for r in covered if r['oracle_iou'] >= .5]
            result[name][mode] = dict(targets=len(filtered), visible=len(visible), absent=len(absent),
                reference_available_visible=len(covered), oracle_valid_seed_visible=sum(r['oracle_iou'] >= .5 for r in visible),
                top1_valid_visible=sum(r['methods'][mode]['iou'] >= .5 for r in visible),
                mean_top1_iou_visible=float(np.mean([r['methods'][mode]['iou'] for r in visible])) if visible else None,
                top1_valid_covered=sum(r['methods'][mode]['iou'] >= .5 for r in covered),
                oracle_possible_covered=len(possible),
                top1_valid_when_possible=sum(r['methods'][mode]['iou'] >= .5 for r in possible),
                policy_accepted_valid=sum(r['methods'][mode]['accepted'] and r['methods'][mode]['iou'] >= .5 for r in visible),
                policy_absent_false_accepts=sum(r['methods'][mode]['accepted'] for r in absent))
    return result


def tint(image, mask, label, color):
    im = image.copy()
    if mask is not None:
        im[mask] = (im[mask]*.5+np.array(color)*.5).astype('uint8')
    im = cv2.resize(im, (400, 225))
    im = np.pad(im, ((28,0),(0,0),(0,0)))
    cv2.putText(im, label, (5,19), cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); (a.output/'qa').mkdir()
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(20260918)
    from detector import Detector
    from identity_guard import Appearance
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    objects = {o['object_id']: o for o in json.loads((BASE/'config/objects.json').read_text())}
    for oid, name in PARTS.items():
        phrase = 'robot gripper' if oid in (1103,1104) else 'robot arm'
        objects[oid] = dict(object_id=oid, class_name=name, grounding_phrase=phrase,
                           detailed_text_prompt=f'a mechanical {phrase}')
    config = PROJECT/'third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    dino = PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'
    weights = PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt'
    bank = SeedBank(a.bank)
    queries = list(selected_queries().items())
    if a.max_frames:
        queries = queries[:a.max_frames]
    write_json(a.output/'run_config.json', dict(schema='astribot.seed_bank_pilot.v1',
        bank=str(a.bank), bank_sha256=sha(a.bank/'bank.json'), embeddings_sha256=sha(a.bank/'embeddings.npz'),
        dino_sha256=sha(dino), sam2_sha256=sha(weights), script_sha256=sha(__file__),
        bank_code_sha256=sha(BASE/'seed_bank.py'), detector_code_sha256=sha(BASE/'detector.py'),
        candidate_floor=.10, per_object_candidate_limit=5, nms_iou=.5,
        score='0.25 detector +0.75 cosine +0.25 (positive cosine - competing identity cosine)',
        ablations=['DINOv2 RGB', 'DINOv2 masked crop', 'mean of both'],
        acceptance='detector>=0.30; bank methods additionally cosine>=0.50 and margin>=0.05; robot side is never auto-accepted',
        split='Exclude entire query episode from all bank references, across cameras; no fitting.',
        limitations=['Biased development frames, not unseen corpus accuracy.',
                     'IoU>=0.5 is a usable-mask ranking proxy, not pure semantic identity accuracy.',
                     'No bank reference falls back to detector ranking, but abstains from bank acceptance.',
                     'Left/right robot ranking is diagnostic only; visual similarity does not prove laterality.'],
        query_frames=[dict(episode_id=k[0],camera=k[1],frame_idx=k[2]) for k,_ in queries]))
    detector = Detector(str(config),str(dino),str(PROJECT/'models/groundingdino/bert-base-uncased'),objects)
    segmenter = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(weights),device='cuda',apply_postprocessing=False))
    encoder = Appearance(); checked = set(); results = []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with (a.output/'candidates.jsonl').open('x') as candfile, (a.output/'queries.jsonl').open('x') as queryfile:
        for number, (key, labels) in enumerate(queries):
            image = read_image(labels[0], checked); h,w = image.shape[:2]
            all_candidates = []
            # Candidate generation is identical for every ranking method and sees no GT mask.
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                segmenter.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for label in labels:
                ids = label.get('merged_object_ids') or [label['object_id']]
                candidates = []
                for oid in ids:
                    proposed,_ = detector.detect(image,[oid],floor=.10,nms_iou=.5)
                    candidates.extend(proposed)
                candidates = sorted(candidates,key=lambda r:-r['confidence'])[:5]
                for d in candidates:
                    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                        masks,scores,_ = segmenter.predict(box=np.array(d['bbox_xyxy']),multimask_output=False)
                    binary = masks[0].astype(bool)
                    if not binary.any():
                        continue
                    rgb,cutout,_,_ = crops(image,binary)
                    all_candidates.append(dict(target_id=label['object_id'],detector_id=d['object_id'],
                        detector_score=d['confidence'],detector_bbox_xyxy=d['bbox_xyxy'],
                        sam_predicted_iou=float(scores[0]),binary=binary,rgb=rgb,cutout=cutout))
            qr = encoder.embed([c['rgb'] for c in all_candidates]).cpu().numpy()
            qm = encoder.embed([c['cutout'] for c in all_candidates]).cpu().numpy()
            for i,c in enumerate(all_candidates):
                c['candidate_id'] = f'Q{number:03d}C{i:03d}'
                c['retrieval'] = {mode:bank.query(qr[i],qm[i],c['target_id'],key[1],exclude_episode=key[0],mode=mode)
                                  for mode in ('rgb','masked','hybrid')}
                c['scores'] = {mode:rank_score(c['detector_score'],r) for mode,r in c['retrieval'].items()}
                c['scores']['detector'] = c['detector_score']
            for label in labels:
                gt = decode(label['mask']); oid = label['object_id']
                candidates = [c for c in all_candidates if c['target_id'] == oid]
                methods = {}; selections = {}
                for mode in ('detector','rgb','masked','hybrid'):
                    ranked = sorted(candidates,key=lambda c:(-c['scores'][mode],c['candidate_id']))
                    chosen = ranked[0] if ranked else None; selections[mode] = chosen
                    binary = chosen['binary'] if chosen else np.zeros((h,w),bool)
                    accepted = chosen is not None and chosen['detector_score'] >= .30
                    if mode != 'detector' and chosen:
                        r = chosen['retrieval'][mode]
                        accepted = accepted and r['similarity'] is not None and r['similarity'] >= .50 and r['margin'] is not None and r['margin'] >= .05 and oid < 1000
                    methods[mode] = dict(candidate_id=chosen['candidate_id'] if chosen else None,
                        ranking=[c['candidate_id'] for c in ranked], accepted=bool(accepted),**metrics(binary,gt))
                has_reference = any(e['episode_id'] != key[0] and e.get('bank_status') == 'active' and
                                    oid in (e.get('merged_object_ids') or [e['object_id']]) for e in bank.entries)
                row = dict(query_id=f'Q{number:03d}_{oid}', sample_id=label['sample_id'],episode_id=key[0],
                    camera=key[1],frame_idx=key[2],object_id=oid,class_name=label['class_name'],
                    task_id=label['task_id'],timestamp=label['timestamp'],gt_visible=bool(gt.any()),
                    source_image_sha256=label['provenance']['source_image_sha256'],
                    gt_source_sha256=label['source_manifest_sha256'],reference_available=has_reference,
                    candidate_count=len(candidates),oracle_iou=max([metrics(c['binary'],gt)['iou'] for c in candidates],default=0),
                    methods=methods)
                results.append(row); queryfile.write(json.dumps(row,allow_nan=False)+'\n')
                for c in candidates:
                    payload = {k:v for k,v in c.items() if k not in ('binary','rgb','cutout')}
                    payload.update(query_id=row['query_id'],episode_id=key[0],camera=key[1],frame_idx=key[2],
                                   **record(c['binary']),reference_metrics=metrics(c['binary'],gt))
                    candfile.write(json.dumps(payload,allow_nan=False)+'\n')
                # Fixed confusion suite and changed rankings are reviewable, including failures.
                changed = methods['detector']['candidate_id'] != methods['hybrid']['candidate_id']
                if oid in (0,1,4,5,10,20,21) or changed:
                    panels = [tint(image,None,f'{row["query_id"]} RGB {key[1]}',(0,0,0)),
                              tint(image,gt,'Accepted reference',(0,220,0))]
                    for mode,color in [('detector',(0,180,255)),('hybrid',(255,180,0))]:
                        chosen = selections[mode]
                        panels.append(tint(image,chosen['binary'] if chosen else None,
                                           f'{mode} IoU={methods[mode]["iou"]:.3f}',color))
                    cv2.imwrite(str(a.output/'qa'/f'{row["query_id"]}.jpg'),np.concatenate(panels,1))
            candfile.flush(); queryfile.flush()
            write_json(a.output/'progress.json',dict(frames=number+1,total_frames=len(queries),targets=len(results),seconds=time.perf_counter()-started))
            print(f'frame {number+1}/{len(queries)} targets={len(results)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    groups = defaultdict(list)
    for r in results:
        groups[f'{r["object_id"]}/{r["camera"]}'].append(r)
    write_json(a.output/'metrics.json',dict(summary=summarize(results),
        by_identity_camera={k:summarize(v)['all'] for k,v in groups.items()},
        frames=len(queries),seconds=time.perf_counter()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        candidates_sha256=sha(a.output/'candidates.jsonl'),queries_sha256=sha(a.output/'queries.jsonl')))
    print(json.dumps(summarize(results),indent=2),flush=True)


if __name__ == '__main__':
    main()
