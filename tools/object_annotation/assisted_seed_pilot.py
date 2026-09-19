"""Bounded mask-guided SAM2 pilot. Only explicitly listed labels are read as seeds.

Predictions are unconfirmed drafts. Scoring is a separate command so references
outside the seed manifest cannot affect the inference path.
"""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

from annotate import PROJECT, QAVideo, sha, write_json
from masks import decode, record

COLORS = {2:(35,220,255), 4:(45,140,255), 10:(220,95,235), 20:(80,65,240), 21:(125,235,65)}


def render(image, rows, caption):
    overlay = image.copy()
    labels = []
    for row in rows:
        if not row['visible']:
            continue
        mask = decode(row['mask']); color = COLORS.get(row['object_id'], (220,220,220))
        overlay[mask] = (overlay[mask]*.55+np.asarray(color)*.45).astype(np.uint8)
        x0,y0,x1,y1 = row['bbox_xyxy']
        cv2.rectangle(overlay,(x0,y0),(x1,y1),color,2)
        score = row.get('confidence')
        label = row['class_name'] + (f' p={score:.2f}' if score is not None else '')
        labels.append((label,x0,y0,y1,color))
    occupied = []
    for label,x0,y0,y1,color in labels:
        (tw,th),baseline = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,1)
        height = th+baseline+6
        left = max(0,min(x0,image.shape[1]-tw-6))
        candidates = [y0-height-4,y1+4]+list(range(0,image.shape[0]-height,height+4))
        for candidate in candidates:
            top = max(0,min(candidate,image.shape[0]-height))
            rect = (left,top,left+tw+6,top+height)
            if all(rect[2]+2<=r[0] or r[2]+2<=rect[0] or rect[3]+2<=r[1] or r[3]+2<=rect[1] for r in occupied):
                break
        occupied.append(rect)
        overlay[top:top+height,left:left+tw+6] //= 3
        cv2.putText(overlay,label,(left+3,top+th+3),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
    canvas = np.concatenate([cv2.resize(image,(640,360)),cv2.resize(overlay,(640,360))],1)
    canvas = np.pad(canvas,((32,0),(0,0),(0,0)))
    cv2.putText(canvas,caption,(6,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()); original = json.loads((a.audit/'manifest.json').read_text())
    chosen = {c['clip_id']: c for c in cfg['clips']}
    samples = [dict(s, source_ui_index=i+1) for i,s in enumerate(original['samples']) if s['clip_id'] in chosen]
    assert len(chosen)==len(cfg['clips']) and samples
    selected_seeds = {idx for c in cfg['clips'] for idx in c['seed_ui_indices']}
    assert all(any(s['source_ui_index']==idx and s['clip_id']==c['clip_id'] for s in samples)
               for c in cfg['clips'] for idx in c['seed_ui_indices'])
    a.output.mkdir(parents=True, exist_ok=False)
    audit = a.output/'audit'; (audit/'human_labels').mkdir(parents=True); (audit/'proposed_labels').mkdir()
    (audit/'images').mkdir()
    for s in samples:
        assert sha(a.audit/s['image'])==s['image_sha256']
        shutil.copy2(a.audit/s['image'], audit/s['image'])
    manifest = {**original, 'samples':samples, 'clips':[c for c in original['clips'] if c['clip_id'] in chosen]}
    write_json(audit/'manifest.json',manifest)
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(20260917)
    checkpoint = PROJECT/cfg['checkpoint']
    predictor = build_sam2_video_predictor(cfg['model_config'],str(checkpoint),device='cuda',
        apply_postprocessing=False,hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    provenance = dict(config=cfg, config_sha256=sha(a.config), checkpoint_sha256=sha(checkpoint),
        audit_manifest_sha256=sha(a.audit/'manifest.json'), script_sha256=sha(__file__),
        direction='independent forward and reverse; offline annotation, not causal detection',
        robot_policy='existing robot masks unchanged; no subtraction or object exclusivity')
    write_json(a.output/'run_config.json',provenance)
    seed_ids, seed_hashes, stats = [], {}, []
    for clip_id, plan in chosen.items():
        clip = sorted([s for s in samples if s['clip_id']==clip_id],key=lambda s:s['frame_idx'])
        cache = a.output/'frames'/clip_id; cache.mkdir(parents=True)
        seeds = defaultdict(list)
        for local,s in enumerate(clip):
            (cache/f'{local:05d}.jpg').symlink_to((audit/s['image']).resolve())
            if s['source_ui_index'] not in selected_seeds:
                continue
            seed_ids.append(s['sample_id'])
            for obj in s['objects']:
                name = f'{s["sample_id"]}__{obj["object_id"]}.json'
                src = a.audit/'human_labels'/name
                row = json.loads(src.read_text())
                assert row['human_confirmed'] and row['source_image_sha256']==s['image_sha256']
                binary = decode(row['mask'])
                assert binary.shape==(s['image_height'],s['image_width'])
                seeds[obj['object_id']].append((local,binary))
                seed_hashes[name] = sha(src)
                shutil.copy2(src,audit/'human_labels'/name)
        objects = clip[0]['objects']; absent = set(plan['visually_absent_object_ids'])
        positive = {oid for oid,values in seeds.items() if any(mask.any() for _,mask in values)}
        assert not positive&absent
        assert positive|absent=={o['object_id'] for o in objects}, 'Missing visual identity or positive mask seed'
        output = {}; confidence = {}; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            for reverse in [False,True]:
                # Independent state prevents forward memory leaking into reverse inference.
                state = predictor.init_state(str(cache),offload_video_to_cpu=True,offload_state_to_cpu=True)
                for oid in sorted(positive):
                    for local,binary in seeds[oid]:
                        predictor.add_new_mask(state,local,oid,binary)
                first = min(local for oid in positive for local,_ in seeds[oid])
                for local,oids,logits in predictor.propagate_in_video(state,start_frame_idx=first,reverse=reverse):
                    if reverse and local>=first:
                        continue
                    output[local] = {oid:binary for oid,binary in zip(oids,(logits[:,0]>0).cpu().numpy())}
                    for obj_idx,oid in enumerate(oids):
                        cached = state['output_dict_per_obj'][obj_idx]
                        current = cached['cond_frame_outputs'].get(local)
                        if current is None:
                            current = cached['non_cond_frame_outputs'][local]
                        confidence[local,oid] = float(current['object_score_logits'].float().sigmoid().reshape(-1)[0])
                del state
        assert set(output)==set(range(len(clip)))
        torch.cuda.synchronize(); seconds = time.perf_counter()-started
        by_frame = []; folder = a.output/f'task_{clip[0]["task_id"]:02d}'/clip[0]['episode_id']; folder.mkdir(parents=True,exist_ok=True)
        path = folder/f'{clip[0]["camera"]}.assisted.jsonl'
        writer = QAVideo(a.output/'qa'/f'{clip_id}.mp4',6)
        contact = []
        with path.open('x') as f:
            for local,s in enumerate(clip):
                rows = []
                for obj in objects:
                    oid = obj['object_id']
                    binary = output[local][oid] if oid in positive else np.zeros((s['image_height'],s['image_width']),bool)
                    row = dict(episode_id=s['episode_id'],task_id=s['task_id'],camera=s['camera'],
                        frame_idx=s['frame_idx'],timestamp=s['timestamp'],sample_id=s['sample_id'],**obj,
                        image_width=s['image_width'],image_height=s['image_height'],
                        track_id=f'{s["episode_id"]}:{s["camera"]}:{oid}',
                        **record(binary),confidence=confidence.get((local,oid)),
                        confidence_semantics='SAM2 presence, not calibrated mask quality; null for visual absence',
                        provenance=dict(mode='visual_absence_review' if oid in absent else 'reviewed_mask_seed_sam2',
                            clip_id=clip_id,seed_frames=[clip[j]['sample_id'] for j,_ in seeds[oid]],
                            seed_manifest='seed_manifest.json'),human_confirmed=False,suspicious_flags=[])
                    f.write(json.dumps(row,allow_nan=False)+'\n'); rows.append(row)
                    if s['source_ui_index'] not in selected_seeds:
                        proposal = dict(sample_id=s['sample_id'],episode_id=s['episode_id'],camera=s['camera'],
                            frame_idx=s['frame_idx'],object_id=oid,instance_id=obj['class_name'].replace(' ','_')+'_1',
                            visibility='visible' if binary.any() else 'out_of_view' if oid in absent else 'unknown',
                            conditions=[],mask=row['mask'],human_confirmed=False,prediction_prefill=True,
                            annotation_method='sam2_reviewed_sparse_seed',source_image_sha256=s['image_sha256'],
                            proposal_id=f'sparse_seed_20260917:{s["sample_id"]}:{oid}',model=cfg['model_config'],
                            notes='Assisted draft; no new human acceptance. Model-empty is unknown unless visually checked.')
                        write_json(audit/'proposed_labels'/f'{s["sample_id"]}__{oid}.json',proposal)
                image = cv2.imread(str(audit/s['image']))
                canvas = render(image,rows,f'{s["camera"]} frame={s["frame_idx"]} | RGB / sparse-mask-seeded SAM2')
                writer.add(canvas)
                cv2.imwrite(str(a.output/'qa'/f'{clip_id}_{local:02d}.jpg'),canvas)
                contact.append(cv2.resize(canvas,(640,196)))
                by_frame.append(rows)
        writer.close()
        cv2.imwrite(str(a.output/'qa'/f'{clip_id}_overview.jpg'),cv2.vconcat(contact))
        stats.append(dict(clip_id=clip_id,frames=len(clip),seconds=seconds,images_per_second=len(clip)/seconds,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,output_sha256=sha(path)))
        print(json.dumps(stats[-1]),flush=True)
        del output, by_frame; gc.collect(); torch.cuda.empty_cache()
    write_json(a.output/'seed_manifest.json',dict(sample_ids=sorted(seed_ids),label_hashes=seed_hashes,
        labels=len(seed_hashes),scoring_exclusion='Every object at every seed frame is excluded for all compared methods'))
    write_json(a.output/'run_complete.json',dict(status='COMPLETE',streams=stats,frames=len(samples),
        seed_frames=len(seed_ids),prediction_frames=len(samples)-len(seed_ids),
        interpretation='Reviewed sparse-seed development pilot, not blind generalization or new human-confirmed labels'))


if __name__=='__main__':
    main()
