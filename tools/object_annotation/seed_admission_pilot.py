"""Frozen-cache admission ablation and explicitly assisted causal SAM2 seeds."""
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

from annotate import PROJECT, BASE, sha, write_json
from masks import decode, record
from seed_bank_recovery import DEFAULT as PARENT, overlap
from seed_bank_transfer import MODEL, WEIGHTS, choose, old_rows, original_frame
from seed_bank_transfer_qa import Video, labelled
from seed_bank import accepted_rows
from seed_bank_pilot import metrics

DEFAULT = PROJECT/'annotations/seed_admission_pilot_20260918'
MODES = ('guarded_replace', 'guarded_all', 'assisted')
NEW = ('new_T1_left_head', 'new_T24_left_head', 'new_T24_left_right_wrist')


def admission(candidate, pool, previous, idx):
    if candidate is None:
        return False, 'no_candidate'
    r = candidate['retrieval']
    if r['similarity'] is None or r['margin'] is None:
        return False, 'no_identity_reference'
    if r['similarity'] < .50 or r['margin'] < .05:
        return False, 'identity_evidence_low'
    if candidate['sam_predicted_iou'] < .80:
        return False, 'sam_boundary_confidence_low'
    for other in pool:
        if other['object_id'] != candidate['object_id'] and other['detector_score'] >= .30:
            if overlap(candidate['binary'], other['binary']) > .70 and other['scores']['masked'] >= candidate['scores']['masked']-.10:
                return False, 'cross_identity_collision'
    if previous is None or idx-previous['frame_idx'] > 15:
        return False, 'await_second_observation'
    if overlap(candidate['binary'], previous['binary']) < .30:
        return False, 'short_window_disagreement'
    return True, 'admitted'


def load_cfg(root):
    cfg = json.loads((root/'run_config.json').read_text())
    for name, digest in cfg['code'].items():
        assert sha(BASE/name) == digest, name
    assert sha(PARENT/'run_config.json') == cfg['parent_config_sha256']
    assert sha(Path(cfg['bank'])/'bank.json') == cfg['bank_sha256']
    return cfg


def prepare(a):
    parent = json.loads((PARENT/'run_config.json').read_text())
    prompts = json.loads(a.prompts.read_text())
    cases = [c for c in parent['cases'] if c['eval_frame'] is not None or c['case_id'] in NEW]
    byid = {c['case_id']:c for c in cases}
    seen = set()
    for p in prompts:
        case = byid[p['case_id']]; key = (p['case_id'],p['frame_idx'],p['object_id'])
        assert key not in seen; seen.add(key)
        assert case['start'] <= p['frame_idx'] <= case['end'] and p['frame_idx'] != case['eval_frame']
        assert p['object_id'] in case['object_ids']
        assert p['status'] in ('visible','not_visible','uncertain')
        assert (p['frame_idx']-case['start']) % 5 == 0
    for case in cases:
        anchors = {p['frame_idx'] for p in prompts if p['case_id']==case['case_id']}
        assert anchors
        assert {(p['frame_idx'],p['object_id']) for p in prompts if p['case_id']==case['case_id']} == {(idx,oid) for idx in anchors for oid in case['object_ids']}
    a.output.mkdir(exist_ok=False)
    cfg = dict(cases=cases, parent=str(PARENT), parent_config_sha256=sha(PARENT/'run_config.json'),
        bank=parent['bank'], bank_sha256=parent['bank_sha256'], models=parent['models'],
        code={name:sha(BASE/name) for name in ('seed_admission_pilot.py','seed_bank_recovery.py','seed_bank_transfer.py','seed_bank.py','masks.py')},
        prompts=prompts, prompt_source_sha256=sha(a.prompts), modes=MODES,
        settings=dict(similarity=.50,margin=.05,sam_iou=.80,collision_iou=.70,collision_score_gap=.10,
                      temporal_iou=.30,max_confirmation_gap=15,stale_frames=30),
        policy='Fixed prior identity gate plus SAM/duplicate/causal confirmation; no threshold tuning on outcomes. Same cached candidates/check times as parent. Assisted is separate, not an automatic result.',
        limits=['Biased development scoring; assisted scoring only five frames after seed.',
                'No new human GT; assistant absence applies only at the observed frame.',
                'No corpus promotion or robot anatomy claims.'])
    write_json(a.output/'run_config.json',cfg)


def seeds(a):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=load_cfg(a.output); byid={c['case_id']:c for c in cfg['cases']}
    target=a.output/a.seed_revision;target.mkdir(exist_ok=False)
    prompts=json.loads(a.prompts.read_text()) if a.prompts else cfg['prompts']
    assert [(p['case_id'],p['frame_idx'],p['object_id']) for p in prompts] == [(p['case_id'],p['frame_idx'],p['object_id']) for p in cfg['prompts']]
    torch.set_num_threads(3);cv2.setNumThreads(1);torch.manual_seed(20260918)
    predictor=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    result=[]; image_key=None
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for p in prompts:
            case=byid[p['case_id']];key=(p['case_id'],p['frame_idx'])
            if key!=image_key:
                image=original_frame(case,p['frame_idx']);image_key=key
                predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                cv2.imwrite(str(target/f'{key[0]}_{key[1]}_rgb.jpg'),image)
            binary=np.zeros(image.shape[:2],bool);scores=[]
            if p['status']=='visible':
                for comp in [p,*p.get('components',[])]:
                    masks,values,_=predictor.predict(box=np.array(comp['box']),
                        point_coords=np.array(comp['points']),point_labels=np.array(comp['point_labels']),multimask_output=False)
                    binary|=masks[0].astype(bool);scores.append(float(values[0]))
                assert binary.any(),p
            row=dict(case_id=p['case_id'],frame_idx=p['frame_idx'],object_id=p['object_id'],prompt=p,
                     **record(binary),sam_predicted_iou=scores,human_confirmed=False,rgb_sha256=sha(target/f'{key[0]}_{key[1]}_rgb.jpg'))
            result.append(row)
            drawing=dict(row,confidence=None,class_name=str(p['object_id']))
            rgb,legend=labelled(image,[drawing],f'{p["case_id"]} frame={p["frame_idx"]} object={p["object_id"]} {p["status"]}')
            cv2.imwrite(str(target/f'{key[0]}_{key[1]}_{p["object_id"]}.jpg'),np.concatenate([rgb,legend],0))
    write_json(target/'seeds.json',result)
    write_json(target/'manifest.json',dict(config_sha256=sha(a.output/'run_config.json'),seed_sha256=sha(target/'seeds.json'),
        model_sha256=sha(WEIGHTS),human_confirmed=False,prompts=prompts))
    print(f'Generated {len(result)} seed proposals: {target}',flush=True)


def run(a):
    from sam2.build_sam import build_sam2_video_predictor
    cfg=load_cfg(a.output);folder=a.output/a.seed_revision
    review=json.loads((folder/'review.json').read_text())
    assert review['decision']=='use_as_assisted_seeds' and review['seed_sha256']==sha(folder/'seeds.json')
    seed_rows=json.loads((folder/'seeds.json').read_text());objects={o['object_id']:o for o in json.loads((BASE/'config/objects.json').read_text())}
    torch.set_num_threads(3);cv2.setNumThreads(1);torch.manual_seed(20260918)
    predictor=build_sam2_video_predictor(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    for ci,case in enumerate(cfg['cases']):
        if ci%a.shards!=a.shard:continue
        dest=a.output/case['case_id'];dest.mkdir(exist_ok=False);parent=PARENT/case['case_id']
        meta=json.loads((parent/'complete.json').read_text());assert sha(parent/'candidates.jsonl')==meta['candidates_sha256']
        assert sha(parent/'checks.json')==meta['checks_sha256']
        candidates=defaultdict(list)
        for line in (parent/'candidates.jsonl').read_text().splitlines():
            r=json.loads(line);candidates[r['frame_idx']].append(r)
        checks={c['frame_idx'] for c in json.loads((parent/'checks.json').read_text())}
        selected={(r['frame_idx'],r['object_id']):r for r in seed_rows if r['case_id']==case['case_id']}
        with gzip.open(parent/'single_bank.jsonl.gz','rt') as f:reference={(r['frame_idx'],r['object_id']):r for r in map(json.loads,f)}
        assert sha(parent/'single_bank.jsonl.gz')==meta['output_sha256']['single_bank']
        events=[];output={m:[] for m in MODES};started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            states={m:predictor.init_state(str(parent/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True) for m in MODES}
            tracks={m:{oid:dict(active=False,last=None,seed=None,verified=None) for oid in case['object_ids']} for m in MODES}
            pending={oid:None for oid in case['object_ids']}
            for idx in range(case['start'],case['end']+1,5):
                local=idx-case['start'];end=min(idx+4,case['end']);pool=[]
                if idx in checks:
                    pool=[dict(r,binary=decode(r['mask'])) for r in candidates[idx]]
                    decisions={}
                    for oid in case['object_ids']:
                        chosen,_=choose([r for r in pool if r['object_id']==oid],'bank_rank',True)
                        allowed,reason=admission(chosen,pool,pending[oid],idx)
                        decisions[oid]=(chosen,allowed,reason)
                        # Consecutive observations must individually pass static gates.
                        static_ok=reason in ('admitted','await_second_observation','short_window_disagreement')
                        pending[oid]=dict(chosen,frame_idx=idx) if chosen and static_ok else None
                for mode in MODES:
                    for oid,t in tracks[mode].items():
                        action='keep';reason='no_check';seed=None
                        if mode=='assisted':
                            if (idx,oid) in selected:
                                proposal=selected[idx,oid]
                                action='seed' if proposal['prompt']['status']=='visible' else 'clear_unknown'
                                reason='assistant_'+proposal['prompt']['status'];seed=proposal
                        elif idx in checks:
                            chosen,allowed,reason=decisions[oid]
                            if mode=='guarded_replace' and idx==case['start']:
                                allowed=chosen is not None;reason='baseline_initialization'
                            if allowed:
                                t['verified']=idx
                                if not t['active'] or t['last'] is None or overlap(t['last'],chosen['binary'])<.25:
                                    action='seed';seed=chosen
                            elif mode=='guarded_all' and t['active'] and idx-t['verified']>=30:
                                action='clear_unknown';reason='unverified_track_expired'
                        if action in ('seed','clear_unknown'):
                            assert idx!=case['eval_frame']
                            if t['active']:predictor.remove_object(states[mode],oid,strict=True,need_output=False)
                            t.update(active=False,last=None,seed=None)
                            if action=='seed':
                                predictor.add_new_mask(states[mode],local,oid,decode(seed['mask']))
                                provenance=dict(kind='assistant_prompt_sam2' if mode=='assisted' else 'admission_gated_candidate',
                                    seed_frame=idx,robot_subtraction=False,human_confirmed=False)
                                if mode=='assisted':provenance.update(seed_sha256=review['seed_sha256'],prompt=seed['prompt'])
                                else:provenance.update(candidate_id=seed['candidate_id'],retrieval=seed['retrieval'],parent_candidate_sha256=meta['candidates_sha256'])
                                t.update(active=True,seed=provenance)
                        if idx in checks or (mode=='assisted' and (idx,oid) in selected):
                            events.append(dict(mode=mode,frame_idx=idx,object_id=oid,action=action,reason=reason,
                                candidate_id=decisions[oid][0]['candidate_id'] if mode!='assisted' and idx in checks and decisions[oid][0] else None))
                    preds={}
                    if any(t['active'] for t in tracks[mode].values()):
                        state=states[mode]
                        for frame,oids,logits in predictor.propagate_in_video(state,start_frame_idx=local,max_frame_num_to_track=end-idx):
                            assert local<=frame<=end-case['start'];preds[frame+case['start']]={}
                            for oid,binary in zip(oids,(logits[:,0]>0).cpu().numpy()):
                                cache=state['output_dict_per_obj'][state['obj_id_to_idx'][oid]]
                                value=cache['cond_frame_outputs'].get(frame)
                                if value is None:value=cache['non_cond_frame_outputs'][frame]
                                preds[frame+case['start']][oid]=(binary,float(value['object_score_logits'].float().sigmoid().reshape(-1)[0]))
                        assert set(preds)==set(range(idx,end+1))
                    for actual in range(idx,end+1):
                        for oid,t in tracks[mode].items():
                            row={k:v for k,v in reference[actual,oid].items() if k in ('episode_id','task_id','camera','timestamp','frame_idx','object_id','class_name','image_width','image_height')}
                            row.update(mode=mode,human_confirmed=False,seed_frame=None,suspicious_flags=[])
                            if oid in preds.get(actual,{}):
                                binary,prob=preds[actual][oid];t['last']=binary
                                row.update(**record(binary),confidence=prob,provenance=t['seed'],seed_frame=t['seed']['seed_frame'],
                                    visibility='predicted_visible' if binary.any() else 'unknown',suspicious_flags=['automatic_draft_not_gt'])
                            else:row.update(mask=None,bbox_xyxy=None,mask_area=None,visible=None,confidence=None,visibility='unknown',suspicious_flags=['no_accepted_seed_unknown'])
                            output[mode].append(row)
                print(f'{case["case_id"]} {end}',flush=True)
            del states
        for mode,rows in output.items():
            with gzip.open(dest/f'{mode}.jsonl.gz','wt') as f:
                for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        write_json(dest/'events.json',events)
        write_json(dest/'complete.json',dict(case=case,seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,config_sha256=sha(a.output/'run_config.json'),
            seed_sha256=review['seed_sha256'],review_sha256=sha(folder/'review.json'),events_sha256=sha(dest/'events.json'),
            output_sha256={m:sha(dest/f'{m}.jsonl.gz') for m in MODES}))
        print(f'COMPLETE {case["case_id"]}',flush=True);del output;gc.collect()


def finish(a):
    cfg=load_cfg(a.output);labels=accepted_rows();results=[];counts=Counter();qa=a.output/'qa';qa.mkdir(exist_ok=False)
    videos=[];cv2.setNumThreads(1)
    for case in cfg['cases']:
        dest=a.output/case['case_id'];meta=json.loads((dest/'complete.json').read_text())
        assert meta['config_sha256']==sha(a.output/'run_config.json')
        assert meta['events_sha256']==sha(dest/'events.json')
        for e in json.loads((dest/'events.json').read_text()):counts[f'{e["mode"]}/{e["action"]}/{e["reason"]}']+=1
        layers={};modes=('single_bank','recovery_bank',*MODES)
        for mode in modes:
            source=dest if mode in MODES else PARENT/case['case_id']
            parent_meta=json.loads((source/'complete.json').read_text());path=source/f'{mode}.jsonl.gz'
            assert sha(path)==parent_meta['output_sha256'][mode]
            rows=defaultdict(list);seen=set()
            with gzip.open(path,'rt') as f:
                for r in map(json.loads,f):
                    key=(r['frame_idx'],r['object_id']);assert key not in seen;seen.add(key)
                    assert r['human_confirmed'] is False
                    if r['mask'] is not None:
                        binary=decode(r['mask']);geom=record(binary)
                        assert geom['bbox_xyxy']==r['bbox_xyxy'] and geom['mask_area']==r['mask_area']
                        assert r['seed_frame']<=r['frame_idx'] and r['seed_frame']!=case['eval_frame']
                    rows[r['frame_idx']].append(r);counts['rows']+=1
            assert seen=={(idx,oid) for idx in range(case['start'],case['end']+1) for oid in case['object_ids']}
            layers[mode]=rows
        if case['eval_frame'] is not None:
            gt=[r for r in labels if r['sample_id']==case['eval_sample'] and r['object_id']<1000 and not r.get('exclude_from_metrics')]
            for r in gt:
                truth=decode(r['mask']);assert np.array_equal(original_frame(case,case['eval_frame']),cv2.imread(r['rgb_path']))
                for mode in modes:
                    pred=next(x for x in layers[mode][case['eval_frame']] if x['object_id']==r['object_id'])
                    mask=decode(pred['mask']) if pred['mask'] else np.zeros_like(truth)
                    results.append(dict(case_id=case['case_id'],camera=case['camera'],mode=mode,object_id=r['object_id'],
                        gt_visible=bool(truth.any()),pred_visible=bool(mask.any()),unknown=pred['visible'] is None,
                        dice=float(2*(mask&truth).sum()/(mask.sum()+truth.sum())) if mask.any() or truth.any() else 0.,**metrics(mask,truth)))
        folder=qa/case['case_id'];folder.mkdir();writer=Video(folder/'comparison.mp4',case['episode']['fps'])
        contacts=set(np.linspace(case['start'],case['end'],5,dtype=int).tolist())
        for idx in range(case['start'],case['end']+1):
            image=cv2.imread(str(PARENT/case['case_id']/'frames'/f'{idx-case["start"]:05d}.jpg'));panels=[]
            for rows,title in [([],f'{case["case_id"]} {idx} RGB'),*[(layers[m][idx],m) for m in modes]]:
                rgb,legend=labelled(image,rows,title)
                legend=np.pad(legend,((0,max(1,len(case['object_ids']))*16-legend.shape[0]),(0,0),(0,0)))
                panels.append(np.concatenate([rgb,legend],0))
            canvas=np.concatenate([np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)],0)
            writer.add(canvas)
            if idx in contacts:cv2.imwrite(str(folder/f'{idx}.jpg'),canvas)
        writer.close();cap=cv2.VideoCapture(str(folder/'comparison.mp4'));n=0
        while True:
            ok,img=cap.read()
            if not ok:break
            assert img.shape==writer.shape;n+=1
        cap.release();assert n==case['end']-case['start']+1
        videos.append(dict(path=str((folder/'comparison.mp4').relative_to(a.output)),frames=n,sha256=sha(folder/'comparison.mp4')))
        print(f'Validated {case["case_id"]}',flush=True)
    summary=[]
    for camera in ('all','head','left_wrist','right_wrist'):
        for mode in ('single_bank','recovery_bank',*MODES):
            rows=[r for r in results if r['mode']==mode and (camera=='all' or r['camera']==camera)]
            pos=[r for r in rows if r['gt_visible']];neg=[r for r in rows if not r['gt_visible']]
            summary.append(dict(camera=camera,mode=mode,visible=len(pos),absent=len(neg),
                iou=float(np.mean([r['iou'] for r in pos])),dice=float(np.mean([r['dice'] for r in pos])),
                valid=sum(r['iou']>=.5 for r in pos),miss=sum(not r['pred_visible'] for r in pos),
                absent_fp=sum(r['pred_visible'] for r in neg),unknown=sum(r['unknown'] for r in rows)))
    write_json(a.output/'scoring.json',dict(summary=summary,rows=results))
    write_json(a.output/'validation.json',dict(status='PASS',counts=dict(counts),videos=videos,semantic_acceptance=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','seeds','run','finish'])
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--prompts',type=Path)
    p.add_argument('--seed-revision',default='seeds_r1');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();assert 0<=a.shard<a.shards
    {'prepare':prepare,'seeds':seeds,'run':run,'finish':finish}[a.phase](a)
