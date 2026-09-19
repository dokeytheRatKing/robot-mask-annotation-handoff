"""Bounded, causal SAM2 recovery comparison with a shared detection schedule."""
import argparse
from collections import Counter, defaultdict
import gc
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from annotate import BASE, PROJECT, sha, write_json
from data import frames
from masks import decode, record
from recovery import Settings, diagnostics, terminate_reason
from seed_bank import SeedBank, accepted_rows, CAMERAS
from seed_bank_pilot import metrics
from seed_bank_transfer import MODEL, WEIGHTS, candidates, choose, old_rows, original_frame
from seed_bank_transfer_qa import Video, labelled

MODES = ('single_detector', 'single_bank', 'recovery_detector', 'recovery_bank')
DEFAULT = PROJECT/'annotations/identity_seed_bank_recovery_20260918_r2'


def overlap(a, b):
    union = int((a | b).sum())
    return float((a & b).sum()/union) if union else 0.


def check_reasons(offset, trackers, eval_frame, actual):
    if offset == 0:
        return ['initial']
    # Hold scoring frames out of automatic initialization, not just human prompts.
    if offset % 5 or actual == eval_frame:
        return []
    reasons = []
    if offset % 15 == 0:
        reasons.append('periodic_identity_check')
    for mode, objects in trackers.items():
        if not mode.startswith('recovery_'):
            continue
        if any(not t['active'] for t in objects.values()):
            reasons.append('unseeded_or_terminated_search')
        if any(t['flags'] for t in objects.values()):
            reasons.append('temporal_or_overlap_warning')
    return sorted(set(reasons))


def action_for(track, candidate, reasons):
    if candidate is None:
        track['misses'] += 1
        reason = terminate_reason(track['flags'], track['empty'], track['misses'], track['presence'], Settings())
        return ('terminate', reason) if track['active'] and reason else ('keep', 'no_candidate_unknown')
    track['misses'] = 0
    if not track['active']:
        return 'seed', 'reacquire' if track['ever_seeded'] else 'first_visible_candidate'
    severe = any(f != 'robot_overlap_warning' for f in track['flags'])
    if severe or track['last'] is None or overlap(track['last'], candidate['binary']) < .25:
        return 'seed', 'candidate_disagreement_or_temporal_anomaly'
    return 'keep', 'candidate_supports_track'


def new_track():
    return dict(active=False, ever_seeded=False, last=None, flags=[], empty=0,
                misses=0, presence=0., seed=None, seed_frame=None)


def prepare(a):
    parent = json.loads((a.parent/'run_config.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = dict(schema='astribot.seed_bank_recovery.v1', parent=str(a.parent.resolve()),
        parent_config_sha256=sha(a.parent/'run_config.json'), bank=parent['bank'],
        bank_sha256=parent['bank_sha256'], modes=list(MODES), cases=parent['cases'],
        models={k: v for k, v in parent.items() if k in ('sam2_sha256','dino_sha256','encoder_sha256')},
        code={name: sha(BASE/name) for name in ('seed_bank_recovery.py','seed_bank_transfer.py',
              'seed_bank.py','recovery.py','identity_guard.py','detector.py')},
        settings=dict(check_grid=5, periodic_frames=15, detector_threshold=.30,
                      disagreement_iou=.25, missing_checks=3, empty_frames=3),
        policy='Causal forward only; shared union-trigger schedule and exact same candidate pool for both recovery arms; eligibility before ranking; no strict bank gate.',
        scoring='Evaluation frames cannot be seed frames. GT masks never used by inference. Query episode excluded from bank.',
        limits=['Small biased development set, not corpus accuracy.',
                'Single-start controls recomputed with the same SAM2 state and candidate implementation.',
                'Track termination is suspected loss, not GT absence; never subtract robot pixels.',
                'No dense temporal GT: event counts are not reentry success or ID-switch rates.'])
    write_json(a.output/'run_config.json', cfg)
    print(f'Prepared {len(cfg["cases"])} cases', flush=True)


def run(a):
    cfg = json.loads((a.output/'run_config.json').read_text())
    for name, digest in cfg['code'].items():
        assert sha(BASE/name) == digest, name
    bank = SeedBank(Path(cfg['bank']))
    assert sha(Path(cfg['bank'])/'bank.json') == cfg['bank_sha256']
    torch.set_num_threads(3); cv2.setNumThreads(1); torch.manual_seed(20260918)
    from detector import Detector
    from identity_guard import Appearance
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    objects = {o['object_id']: o for o in json.loads((BASE/'config/objects.json').read_text())}
    detector = Detector(str(PROJECT/'third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
        str(PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'),
        str(PROJECT/'models/groundingdino/bert-base-uncased'), objects)
    segmenter = SAM2ImagePredictor(build_sam2(MODEL, str(WEIGHTS), device='cuda', apply_postprocessing=False))
    encoder = Appearance()
    predictor = build_sam2_video_predictor(MODEL, str(WEIGHTS), device='cuda', apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    for i, case in enumerate(cfg['cases']):
        if i % a.shards != a.shard:
            continue
        directory = a.output/case['case_id']
        if a.skip_complete and (directory/'complete.json').exists():
            meta=json.loads((directory/'complete.json').read_text())
            assert meta['case']==case
            for mode in MODES: assert sha(directory/f'{mode}.jsonl.gz')==meta['output_sha256'][mode]
            print(f'SKIP verified {case["case_id"]}',flush=True)
            continue
        directory.mkdir(exist_ok=False)
        cache = directory/'frames'; cache.mkdir()
        images = {}; timestamps = {}; ep = case['episode']; cam = case['camera']; ids = case['object_ids']
        started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for idx, ts, image in frames(ep, cam):
            if idx < case['start']: continue
            if idx > case['end']: break
            images[idx] = image; timestamps[idx] = ts
            assert cv2.imwrite(str(cache/f'{idx-case["start"]:05d}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY,98])
        assert len(images) == case['end']-case['start']+1
        robots, robot_source = old_rows(ep, cam, case['start'], case['end'], 'robot')
        tracks = {mode: {oid: new_track() for oid in ids} for mode in MODES}
        output = {mode: defaultdict(list) for mode in MODES}; events = []; checks = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), (directory/'candidates.jsonl').open('x') as pool_file:
            states = {mode: predictor.init_state(str(cache), offload_video_to_cpu=True, offload_state_to_cpu=True) for mode in MODES}
            for idx in range(case['start'], case['end']+1, 5):
                local = idx-case['start']; end = min(idx+4, case['end'])
                reasons = check_reasons(local, tracks, case['eval_frame'], idx)
                if reasons:
                    # GroundingDINO deformable attention requires FP32; candidates
                    # enables BF16 locally only for the SAM2 image calls.
                    with torch.autocast('cuda', enabled=False):
                        pool = candidates(images[idx], ids, ep['episode_id'], cam, idx, detector, segmenter, encoder, bank)
                    for item in pool:
                        payload = {k:v for k,v in item.items() if k not in ('binary','rgb','cutout')}
                        pool_file.write(json.dumps(dict(payload,frame_idx=idx,**record(item['binary'])), allow_nan=False)+'\n')
                    checks.append(dict(frame_idx=idx, reasons=reasons, candidate_count=len(pool)))
                    for mode in MODES:
                        if mode.startswith('single_') and local:
                            continue
                        for oid in ids:
                            track = tracks[mode][oid]
                            chosen, selected_reason = choose([x for x in pool if x['object_id']==oid],
                                'bank_rank' if mode.endswith('bank') else 'detector', eligible_first=True)
                            action, reason = action_for(track, chosen, reasons)
                            events.append(dict(mode=mode, object_id=oid, frame_idx=idx, action=action, reason=reason,
                                selection_reason=selected_reason, candidate_id=chosen['candidate_id'] if chosen else None,
                                previous_seed_frame=track['seed_frame'], flags=list(track['flags']), misses=track['misses']))
                            if action in ('seed','terminate') and track['active']:
                                predictor.remove_object(states[mode], oid, strict=True, need_output=False)
                                track['active'] = False
                            if action == 'seed':
                                assert idx != case['eval_frame']
                                predictor.add_new_mask(states[mode], local, oid, chosen['binary'])
                                track.update(active=True,ever_seeded=True,seed=chosen,seed_frame=idx,empty=0)
                            elif action == 'terminate':
                                track.update(last=None, seed=None, seed_frame=None, presence=0.)
                for mode in MODES:
                    for t in tracks[mode].values(): t['flags'] = []
                    predictions = {}
                    if any(t['active'] for t in tracks[mode].values()):
                        state = states[mode]
                        for frame, oids, logits in predictor.propagate_in_video(state, start_frame_idx=local,
                                max_frame_num_to_track=end-idx):
                            assert local <= frame <= end-case['start']
                            predictions[frame+case['start']] = {}
                            for oid, binary in zip(oids, (logits[:,0]>0).cpu().numpy()):
                                obj_idx = state['obj_id_to_idx'][oid]
                                cached = state['output_dict_per_obj'][obj_idx]
                                value = cached['cond_frame_outputs'].get(frame)
                                if value is None: value = cached['non_cond_frame_outputs'][frame]
                                prob = float(value['object_score_logits'].float().sigmoid().reshape(-1)[0])
                                predictions[frame+case['start']][oid] = (binary, prob)
                        assert set(predictions) == set(range(idx,end+1))
                    for actual in range(idx,end+1):
                        binary = {oid: pair[0] for oid,pair in predictions.get(actual,{}).items()}
                        previous = {oid:t['last'] for oid,t in tracks[mode].items() if t['last'] is not None}
                        warnings = diagnostics(binary, previous, decode(robots[actual][0]['mask']), Settings())
                        for oid in ids:
                            track = tracks[mode][oid]
                            row = dict(episode_id=ep['episode_id'],task_id=ep['task_id'],camera=cam,frame_idx=actual,
                                timestamp=timestamps[actual],object_id=oid,class_name=objects[oid]['class_name'],mode=mode,
                                human_confirmed=False,image_width=images[actual].shape[1],image_height=images[actual].shape[0],
                                seed_frame=track['seed_frame'])
                            if oid not in binary:
                                row.update(mask=None,bbox_xyxy=None,mask_area=None,visible=None,confidence=None,
                                    visibility='unknown',suspicious_flags=['terminated_unknown' if track['ever_seeded'] else 'unseeded_unknown'])
                            else:
                                m, prob = predictions[actual][oid]
                                track['empty'] = track['empty']+1 if not m.any() else 0
                                track['last'] = m; track['presence'] = prob
                                track['flags'] = sorted(set(track['flags'])|set(warnings.get(oid,[])))
                                seed = track['seed']
                                row.update(**record(m),confidence=prob,visibility='predicted_visible' if m.any() else 'unknown',
                                    suspicious_flags=warnings.get(oid,[]),provenance=dict(kind='automatic_recovery_not_gt',
                                    candidate_id=seed['candidate_id'],detector_score=seed['detector_score'],bank=seed['retrieval'],
                                    bank_sha256=cfg['bank_sha256'],robot_subtraction=False))
                            output[mode][actual].append(row)
                print(f'{case["case_id"]} frame {end}', flush=True)
            del states
        for mode in MODES:
            with gzip.open(directory/f'{mode}.jsonl.gz','wt') as out:
                for idx in sorted(output[mode]):
                    for row in output[mode][idx]: out.write(json.dumps(row,allow_nan=False)+'\n')
        write_json(directory/'events.json',events); write_json(directory/'checks.json',checks)
        write_json(directory/'complete.json',dict(status='COMPLETE',case=case,frame_count=len(images),
            seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            robot_source=robot_source,events_sha256=sha(directory/'events.json'),checks_sha256=sha(directory/'checks.json'),
            candidates_sha256=sha(directory/'candidates.jsonl'),
            output_sha256={mode:sha(directory/f'{mode}.jsonl.gz') for mode in MODES}))
        print(f'COMPLETE {case["case_id"]}', flush=True)
        del output, images, tracks, robots
        gc.collect()


def finish(a):
    cfg = json.loads((a.output/'run_config.json').read_text()); accepted = accepted_rows()
    scores = []; aggregate_events = Counter(); manifest = []; total_rows = 0
    qa = a.output/'qa'; qa.mkdir(exist_ok=False); cv2.setNumThreads(1)
    for case in cfg['cases']:
        directory = a.output/case['case_id']; meta = json.loads((directory/'complete.json').read_text())
        assert sha(directory/'events.json') == meta['events_sha256']
        events = json.loads((directory/'events.json').read_text())
        for event in events: aggregate_events[f'{event["mode"]}/{event["action"]}/{event["reason"]}'] += 1
        layers = {}; shape = None
        for mode in MODES:
            assert sha(directory/f'{mode}.jsonl.gz') == meta['output_sha256'][mode]
            layers[mode] = defaultdict(list); seen = set()
            with gzip.open(directory/f'{mode}.jsonl.gz','rt') as f:
                for row in map(json.loads,f):
                    key = (row['frame_idx'],row['object_id']); assert key not in seen; seen.add(key)
                    assert row['episode_id']==case['episode']['episode_id'] and row['camera']==case['camera']
                    assert row['human_confirmed'] is False
                    if row['mask'] is not None:
                        m = decode(row['mask']); r = record(m); shape = m.shape
                        assert r['bbox_xyxy']==row['bbox_xyxy'] and r['mask_area']==row['mask_area'] and r['visible']==row['visible']
                        assert row['seed_frame'] <= row['frame_idx'] and row['seed_frame'] != case['eval_frame']
                    layers[mode][row['frame_idx']].append(row); total_rows += 1
            assert seen == {(idx,oid) for idx in range(case['start'],case['end']+1) for oid in case['object_ids']}
        if case['eval_frame'] is not None:
            labels = [r for r in accepted if r['sample_id']==case['eval_sample'] and r['object_id']<1000 and not r.get('exclude_from_metrics')]
            for label in labels:
                gt = decode(label['mask']); assert np.array_equal(original_frame(case,case['eval_frame']),cv2.imread(label['rgb_path']))
                for mode in MODES:
                    row = next(r for r in layers[mode][case['eval_frame']] if r['object_id']==label['object_id'])
                    pred = decode(row['mask']) if row['mask'] else np.zeros_like(gt)
                    scores.append(dict(case_id=case['case_id'],camera=case['camera'],object_id=label['object_id'],mode=mode,
                        gt_visible=bool(gt.any()),pred_visible=bool(pred.any()),
                        dice=float(2*(pred&gt).sum()/(pred.sum()+gt.sum())) if pred.any() or gt.any() else 0.,
                        **metrics(pred,gt)))
        dest = qa/case['case_id']; dest.mkdir(); writer = Video(dest/'comparison.mp4',case['episode']['fps'])
        robot,_ = old_rows(case['episode'],case['camera'],case['start'],case['end'],'robot')
        contacts = set(np.linspace(case['start'],case['end'],5,dtype=int).tolist())
        if case['eval_frame'] is not None: contacts.add(case['eval_frame'])
        for idx,ts,image in frames(case['episode'],case['camera']):
            if idx<case['start']: continue
            if idx>case['end']: break
            cells=[]
            for rows,title in [([],f'{case["case_id"]} {idx} RGB'),(robot[idx],'Robot QA only'),*[(layers[m][idx],m) for m in MODES]]:
                rgb,legend=labelled(image,rows,title)
                legend=np.pad(legend,((0,max(1,len(case['object_ids']))*16-legend.shape[0]),(0,0),(0,0)))
                cells.append(np.concatenate([rgb,legend],0))
                for r in rows:
                    if r.get('mode'): assert r['timestamp']==ts
            canvas=np.concatenate([np.concatenate(cells[:3],1),np.concatenate(cells[3:],1)],0)
            writer.add(canvas)
            if idx in contacts: assert cv2.imwrite(str(dest/f'qa_{idx}.jpg'),canvas)
        writer.close(); cap=cv2.VideoCapture(str(dest/'comparison.mp4')); count=0
        while True:
            ok,image=cap.read()
            if not ok: break
            assert image.shape==writer.shape;count+=1
        cap.release();assert count==case['end']-case['start']+1
        manifest.append(dict(case_id=case['case_id'],frames=count,path=str((dest/'comparison.mp4').relative_to(a.output)),sha256=sha(dest/'comparison.mp4')))
        print(f'Validated {case["case_id"]}',flush=True)
    summary=[]
    for camera in ('all',*CAMERAS):
        for mode in MODES:
            rows=[r for r in scores if r['mode']==mode and (camera=='all' or r['camera']==camera)]
            pos=[r for r in rows if r['gt_visible']];neg=[r for r in rows if not r['gt_visible']]
            summary.append(dict(camera=camera,mode=mode,visible=len(pos),absent=len(neg),
                mean_visible_iou=float(np.mean([r['iou'] for r in pos])) if pos else None,
                mean_visible_dice=float(np.mean([r['dice'] for r in pos])) if pos else None,
                valid_visible=sum(r['iou']>=.5 for r in pos),missed_visible=sum(not r['pred_visible'] for r in pos),
                absent_false_positive=sum(r['pred_visible'] for r in neg)))
    write_json(a.output/'scoring.json',dict(summary=summary,rows=scores))
    write_json(a.output/'validation.json',dict(status='PASS',rows=total_rows,videos=manifest,
        events=dict(aggregate_events),config_sha256=sha(a.output/'run_config.json'),
        note='Structural validation is not semantic acceptance; no dense GT recovery-rate claims.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','run','finish'])
    parser.add_argument('--parent',type=Path,default=PROJECT/'annotations/identity_seed_bank_transfer_20260918_r2')
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--skip-complete',action='store_true')
    args=parser.parse_args();assert 0<=args.shard<args.shards
    {'prepare':prepare,'run':run,'finish':finish}[args.phase](args)
