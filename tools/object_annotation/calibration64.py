"""Prepare a fixed 64-episode RGB/mask subset for action-metric calibration.

Only data preparation. No model calibration, latent mapping, or training.
"""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from annotate import BASE, PROJECT, sha, write_json
from data import scan, frames
from full_episode_reentry import rows_at
from masks import decode, encode, record
from recovery import Settings, diagnostics
from seed_bank import accepted_rows
from seed_bank_transfer import MODEL, WEIGHTS
from seed_bank_transfer_qa import labelled, Video

ROOT=PROJECT/'annotations/action_metric_calibration64_20260919'
DATA=PROJECT/'datasets/lerobot/astribot_full_v21_rgb_h264'
SEED=20260919
CAMERAS=['head','left_wrist','right_wrist']


def stable_seed(key):
    return int(hashlib.sha256(f'{SEED}/{key}'.encode()).hexdigest()[:8],16)


def read(root,name):return json.loads((root/name).read_text(encoding='utf-8'))


def prepare(a):
    a.root.mkdir(exist_ok=False);eps={e['episode_id']:e for e in scan(DATA)};refs=accepted_rows()
    groups=defaultdict(list)
    for r in refs:
        if eps[r['episode_id']]['episode_index']<5223 and not r.get('exclude_from_metrics'):
            groups[(r['episode_id'],r['camera'],r['frame_idx'])].append(r)
    byep=defaultdict(list)
    for k,rr in groups.items():
        if rr[0]['source_set']=='accepted87':byep[k[0]].append((k,rr))
    assert len(byep)==62
    # Prefer one episode per unit, retain all 29 tasks, balance cameras among alternatives.
    cams=Counter();selected=[]
    for eid in sorted(byep,key=lambda e:(len(byep[e]),stable_seed(e))):
        k,rr=min(byep[eid],key=lambda item:(cams[item[0][1]],
            -sum(r['visible'] is True for r in item[1] if r['object_id']<1000),stable_seed(str(item[0]))))
        selected.append((k,rr));cams[k[1]]+=1
    # Two new episodes with dense accepted object masks; no fabricated robot labels.
    for eid in ['episode_001608','episode_002478']:
        opts=[(k,rr) for k,rr in groups.items() if k[0]==eid]
        k,rr=min(opts,key=lambda item:(-sum(r['visible'] is True for r in item[1]),
            cams[item[0][1]],abs(item[0][2]-eps[eid]['frames']//2),stable_seed(str(item[0]))))
        selected.append((k,rr));cams[k[1]]+=1
    selected.sort(key=lambda x:(eps[x[0][0]]['task_id'],stable_seed(str(x[0]))))
    assert len(selected)==len({k[0] for k,_ in selected})==64
    # Select 16 diagnostic episodes across different tasks, keeping every task in fit.
    counts=Counter(eps[k[0]]['task_id'] for k,_ in selected);diag=set();diag_tasks=set()
    for k,_ in sorted(selected,key=lambda x:stable_seed('diag/'+x[0][0])):
        task=eps[k[0]]['task_id']
        if task not in diag_tasks and counts[task]>=2:
            diag.add(k[0]);diag_tasks.add(task);counts[task]-=1
            if len(diag)==16:break
    assert len(diag)==16
    hc=Counter();cs=[]
    for number,(k,rr) in enumerate(selected):
        eid,cam,anchor=k;ep=eps[eid];task=ep['task_id']
        # Round-robin horizon allocation is fixed before viewing predictions.
        horizon=[16,32,64][number%3];length=17+horizon
        start=max(0,min(anchor-16-horizon//2,ep['frames']-length));end=start+length-1
        cid=f'C{number:03d}_T{task:02d}_{eid}_{cam}'
        ids=sorted({r['object_id'] for r in rr});assert all(i<1000 or 1100<=i<=1104 for i in ids)
        out=a.root/'clips'/cid;out.mkdir(parents=True);(out/'rgb').mkdir();(out/'sam_frames').mkdir()
        timestamps=[];hashes=[];rgb_hashes=[]
        seeds=[r for (ee,cc,ii),rs in groups.items() if ee==eid and cc==cam and start<=ii<=end for r in rs if r['object_id'] in ids]
        assert any(r['frame_idx']==anchor for r in seeds)
        for idx,ts,image in frames(ep,cam):
            if idx<start:continue
            if idx>end:break
            local=idx-start;p=out/'rgb'/f'{local:05d}.png';q=out/'sam_frames'/f'{local:05d}.jpg'
            assert cv2.imwrite(str(p),image,[cv2.IMWRITE_PNG_COMPRESSION,3])
            assert cv2.imwrite(str(q),image,[cv2.IMWRITE_JPEG_QUALITY,95])
            for r in seeds:
                if r['frame_idx']==idx:
                    reference=cv2.imread(r['rgb_path']);assert np.array_equal(reference,image),(cid,idx)
                    assert sha(r['rgb_path'])==r['provenance']['source_image_sha256']
            timestamps.append(ts);hashes.append(sha(q));rgb_hashes.append(sha(p))
        assert len(timestamps)==length
        table=pq.read_table(ep['parquet']).slice(start,length);pq.write_table(table,out/'state_action.parquet')
        aa=np.asarray(table['action'].to_pylist(),np.float32);ss=np.asarray(table['observation.state'].to_pylist(),np.float32)
        assert aa.shape==(length,20) and ss.shape==(length,25) and np.isfinite(aa).all() and np.isfinite(ss).all()
        times={key:np.asarray(table[key].to_numpy(),np.float64)+ep['source_timestamp_start'] for key in
            ['timestamp.source','timestamp.state','timestamp.action','timestamp.images.head','timestamp.images.left','timestamp.images.right']}
        np.savez_compressed(out/'state_action.npz',action_absolute_20d=aa,state_raw_25d=ss,
            frame_idx=np.arange(start,end+1),action_valid_dimensions=np.ones(20,bool),**times)
        write_json(out/'accepted_seeds.json',seeds)
        pred_start=start+17
        c=dict(clip_id=cid,episode=ep,camera=cam,task_id=task,side=ep['side'],split='C_diag' if eid in diag else 'C_fit',
            in_H=False,start=start,end=end,frame_count=length,anchor_frame=anchor,object_ids=[i for i in ids if i<1000],
            robot_part_ids=[i for i in ids if i>=1100],seed_frames=sorted({r['frame_idx'] for r in seeds}),
            context_frames=list(range(start,start+17)),prediction_frames=list(range(pred_start,end+1)),
            source_horizon_frames=horizon,nominal_horizon_seconds=horizon/30,
            physical_horizon_seconds=timestamps[-1]-timestamps[16],timestamps=timestamps,
            prediction_timestamps=timestamps[17:],reference_timestamp=timestamps[16],
            source_rgb_sha256=sha(ep['cameras'][cam]['video']),source_parquet_sha256=sha(ep['parquet']),
            original_timestamp_start=ep['source_timestamp_start'],rgb_sha256=rgb_hashes,sam_jpeg_sha256=hashes,
            shape=list(image.shape),seed_sha256=sha(out/'accepted_seeds.json'),state_action_sha256=sha(out/'state_action.npz'),
            parquet_sha256=sha(out/'state_action.parquet'),native_binding=dict(status='pending_remote_adapter',
                selected_action_chunk=None,condition_video_group=None,causal_token_ids=None,vae_temporal_support=None,
                action_tau=None,tau_bin=None,conditioning_noise_level=None),
            random_seeds=dict(sampling=SEED,action_noise=stable_seed(cid+'/action'),video_noise=stable_seed(cid+'/video'),
                probes=[stable_seed(cid+f'/probe/{i}') for i in range(4)]))
        cs.append(c);print('PREPARED',cid,length,flush=True)
    # H is an episode-isolated subset of C_fit, spread over tasks/cameras/horizons.
    tasks=Counter();cam_count=Counter();hz_count=Counter();remaining=[c for c in cs if c['split']=='C_fit']
    for _ in range(16):
        c=min(remaining,key=lambda c:(tasks[c['task_id']],cam_count[c['camera']],hz_count[c['source_horizon_frames']],stable_seed(c['clip_id']+'/H')))
        c['in_H']=True;remaining.remove(c);tasks[c['task_id']]+=1;cam_count[c['camera']]+=1;hz_count[c['source_horizon_frames']]+=1
    info=read(DATA,'meta/info.json')
    config=dict(schema='astribot.action_metric_calibration64.v1',clips=cs,seed=SEED,
        requested_clips=64,idea_sha256=sha(PROJECT/'docs/task_centric_world_modeling_idea_summary.md'),
        source_split=dict(train=[0,5223],held_out=[5223,5803]),
        mask_policy='Task objects and separate robot parts. Only reviewed reliable object union + reliable EE contributes ROI. Unknown never becomes confirmed absence.',
        selection='All 62 training episodes in accepted87, one camera each; two dense accepted420 episodes. Biased reviewed pool, not a random corpus sample.',
        propagation='Nearest accepted seed per identity, forward/backward offline SAM2; GT-free frame guards, then assistant context review. Labels may use future frames; model inputs must not.',
        roi_margin_pixels_at_640x360=4,roi_policy='Dilate reviewed foreground locally; unknown foreground contributes zero. Arms excluded from task ROI. No robot subtraction.',
        action=dict(representation='absolute',dimension=20,names=info['features']['action']['names'],normalized=False,padding=0),
        state=dict(dimension=25,names=info['features']['observation.state']['names'],normalized=False),
        binding='Physical source windows only. Remote native temporal VAE support/action groups/tau remain null; never infer them from source indices.',
        sam2=dict(config=MODEL,weights=str(WEIGHTS),sha256=sha(WEIGHTS)),
        code_hashes={n:sha(BASE/n) for n in ['calibration64.py','masks.py','data.py','recovery.py']},
        bank_usage='Existing accepted masks are identity seeds. No global bank hard-veto; no new model training.')
    write_json(a.root/'manifest.json',config)
    write_json(a.root/'splits.json',{k:[c['clip_id'] for c in cs if (c['in_H'] if k=='H' else c['split']==k)] for k in ['C_fit','C_diag','H']})
    print('SPLITS',Counter(c['split'] for c in cs),'CAMERAS',Counter(c['camera'] for c in cs),'HORIZONS',Counter(c['source_horizon_frames'] for c in cs),flush=True)


def load(root):
    cfg=read(root,'manifest.json');assert sha(WEIGHTS)==cfg['sam2']['sha256']
    for n,h in cfg['code_hashes'].items():assert sha(BASE/n)==h,n
    return cfg


def propagate(predictor,c,path,seeds):
    """Independent nearest-seed intervals, deterministic offline bidirectional labeling."""
    byid=defaultdict(list)
    for r in seeds:byid[r['object_id']].append(r)
    allpred={};annotations={};segments=defaultdict(list)
    for oid,rr in byid.items():
        rr.sort(key=lambda r:r['frame_idx']);allpred[oid]={};annotations[oid]={}
        for j,r in enumerate(rr):
            s=r['frame_idx'];left=c['start'] if j==0 else (rr[j-1]['frame_idx']+s)//2+1
            right=c['end'] if j==len(rr)-1 else (s+rr[j+1]['frame_idx'])//2
            for idx in range(left,right+1):annotations[oid][idx]=r
            if r['visible'] is not True:continue
            segments[(s,left,right)].append(r)
        assert set(annotations[oid])==set(range(c['start'],c['end']+1))
    for (s,left,right),rr in sorted(segments.items()):
        for reverse,end in [(True,left),(False,right)]:
            state=predictor.init_state(str(path/'sam_frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            for r in rr:predictor.add_new_mask(state,s-c['start'],r['object_id'],decode(r['mask']))
            for local,oids,logits in predictor.propagate_in_video(state,start_frame_idx=s-c['start'],
                    max_frame_num_to_track=abs(end-s),reverse=reverse):
                idx=local+c['start'];assert left<=idx<=right and set(oids)=={r['object_id'] for r in rr}
                for pos,oid in enumerate(oids):
                    cache=state['output_dict_per_obj'][state['obj_id_to_idx'][oid]];v=cache['cond_frame_outputs'].get(local)
                    if v is None:v=cache['non_cond_frame_outputs'][local]
                    allpred[oid][idx]=((logits[pos,0]>0).cpu().numpy(),float(v['object_score_logits'].float().sigmoid().item()))
            del state
    return allpred,annotations


def run(a):
    from sam2.build_sam import build_sam2_video_predictor
    cfg=load(a.root);predictor=build_sam2_video_predictor(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    for ci,c in enumerate(cfg['clips']):
        if ci%a.shards!=a.shard:continue
        out=a.root/'clips'/c['clip_id'];dest=out/'draft';dest.mkdir(exist_ok=False)
        assert sha(out/'accepted_seeds.json')==c['seed_sha256'];seeds=read(out,'accepted_seeds.json')
        for j,h in enumerate(c['sam_jpeg_sha256']):assert sha(out/'sam_frames'/f'{j:05d}.jpg')==h
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred,annotations=propagate(predictor,c,out,seeds)
        previous={};counts=Counter()
        with gzip.open(dest/'objects.jsonl.gz','wt',encoding='utf-8') as objects,gzip.open(dest/'robot_parts.jsonl.gz','wt',encoding='utf-8') as robot:
            for idx in range(c['start'],c['end']+1):
                masks={oid:p[idx][0] for oid,p in pred.items() if idx in p}
                flags=diagnostics({oid:m for oid,m in masks.items() if oid<1000},{oid:m for oid,m in previous.items() if oid<1000},None,Settings())
                for oid,anns in sorted(annotations.items()):
                    seed=anns[idx];isexact=idx==seed['frame_idx'];r=copy.deepcopy(seed)
                    r.update(clip_id=c['clip_id'],episode_id=c['episode']['episode_id'],task_id=c['task_id'],camera=c['camera'],
                        frame_idx=idx,timestamp=c['timestamps'][idx-c['start']],seed_frame=seed['frame_idx'],
                        human_confirmed=isexact,mask=None,mask_area=None,bbox_xyxy=None,visible=None,visibility='unknown',confidence=None,
                        mask_status='unknown',training_eligible=False,suspicious_flags=flags.get(oid,[]),
                        provenance=dict(kind='accepted_seed' if isexact else 'offline_sam2_propagation',seed_sha256=c['seed_sha256'],
                            seed_frame=seed['frame_idx'],source_provenance=seed['provenance'],future_seed=seed['frame_idx']>idx,
                            labels_only=True,source_image_sha256=c['rgb_sha256'][idx-c['start']]))
                    if isexact:
                        for key in ['mask','mask_area','bbox_xyxy','visible','visibility','confidence','mask_status']:
                            if key in seed:r[key]=seed[key]
                        r['mask_status']='human_confirmed' if seed['visible'] is not None else 'unknown'
                        r['training_eligible']=seed['visible'] is True
                    elif oid in masks and masks[oid].any():
                        r.update(**record(masks[oid]),confidence=pred[oid][idx][1],visibility='predicted_visible')
                    for key in ['rgb_path','source_manifest','source_manifest_sha256','sample_id']:r.pop(key,None)
                    counts['rows']+=1;counts['positive']+=r['visible'] is True;counts['exact_seed']+=isexact
                    (objects if oid<1000 else robot).write(json.dumps(r,allow_nan=False)+'\n')
                previous=masks
        write_json(dest/'complete.json',dict(manifest_sha256=sha(a.root/'manifest.json'),
            seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            hashes={k:sha(dest/k) for k in ['objects.jsonl.gz','robot_parts.jsonl.gz']},counts=dict(counts),semantic_acceptance=False))
        print('COMPLETE',c['clip_id'],dict(counts),flush=True)


def qa(a):
    cfg=load(a.root);dest=a.root/'qa';dest.mkdir(exist_ok=True)
    for ci,c in enumerate(cfg['clips']):
        if ci%a.shards!=a.shard:continue
        out=a.root/'clips'/c['clip_id'];meta=read(out/'draft','complete.json')
        for n,h in meta['hashes'].items():assert sha(out/'draft'/n)==h
        objects=rows_at(out/'draft/objects.jsonl.gz');robot=rows_at(out/'draft/robot_parts.jsonl.gz')
        checks=sorted(set(np.linspace(c['start'],c['end'],5,dtype=int).tolist()+[c['anchor_frame']]))
        tiles=[]
        for idx in checks:
            im=cv2.imread(str(out/'rgb'/f'{idx-c["start"]:05d}.png'));cells=[]
            for rows,label in [(objects[idx],f'{c["clip_id"][:8]} {c["camera"]} objects f{idx}'),
                (robot.get(idx,[]),f'robot (EE only for ROI) f{idx}')]:
                view,legend=labelled(im,rows,label)
                legend=np.pad(legend,((0,max(0,80-legend.shape[0])),(0,0),(0,0)))
                cells.append(np.concatenate([view,legend],0))
            tiles.append(np.concatenate(cells,1))
        cv2.imwrite(str(dest/f'{c["clip_id"]}.jpg'),np.concatenate(tiles,0))
        write_json(dest/f'{c["clip_id"]}.json',dict(checks=checks,source_hashes=meta['hashes']))
        print('QA',c['clip_id'],checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','qa']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(SEED);globals()[a.phase](a)
