"""Versioned, assistant-reviewed local repairs of frozen SAM2 object tracks."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import time
import os

import cv2
import numpy as np
import torch

from annotate import BASE, PROJECT, sha, write_json
from full_episode_reentry import rows_at, load as load_parent
from masks import decode, record
from recovery import Settings, diagnostics
from seed_bank_transfer import MODEL, WEIGHTS
from seed_bank_transfer_qa import Video, labelled

PARENT=PROJECT/'annotations/full_episode_reentry_20260918'
DEFAULT=PROJECT/'annotations/semantic_interval_repair_20260918'
SPEC=BASE/'config/semantic_interval_repair_20260918.json'


def inspect(a):
    load_parent(PARENT)
    a.root.mkdir(exist_ok=False)
    out=a.root/'inspection';out.mkdir()
    wanted={'head':[0,15,30,60,90,120,180,240,300,360,420,480,540],
            'right_wrist':[630,640,650,660,668,680,700,720,740,760,780,
                           880,900,920,940,960,980,1000,1020,1040,1080,1120,1140]}
    for cam,indices in wanted.items():
        case=f'episode_002503_{cam}';rows=rows_at(PARENT/case/'recovery/objects.jsonl.gz')
        for page in range(0,len(indices),6):
            tiles=[]
            for idx in indices[page:page+6]:
                image=cv2.imread(str(PARENT/case/'frames'/f'{idx:05d}.jpg'));assert image is not None
                rgb,_=labelled(image,[],f'{cam} RGB {idx}')
                pred,_=labelled(image,rows[idx],f'Previous recovery {idx}')
                tiles.append(np.concatenate([rgb,pred],1))
            cv2.imwrite(str(out/f'{cam}_{page//6}.jpg'),np.concatenate(tiles,0))
    print('Inspection sheets ready',flush=True)


def prepare(a):
    parent=load_parent(PARENT);spec=json.loads(SPEC.read_text())
    sources={c['case_id']:c for c in parent['cases']};out=a.root/'config.json';assert not out.exists()
    hashes={}
    for c in spec['cases']:
        case=sources[c['source_case']];folder=a.root/c['case_id'];folder.mkdir()
        cache=folder/'frames';cache.mkdir();start,end,oid=c['start'],c['end'],c['object_id']
        assert 0<=start<=end<case['episode']['frames'] and oid in case['object_ids']
        assert len({p['frame_idx'] for p in c['prompts']})==len(c['prompts'])
        assert all(start<=p['frame_idx']<=end for p in c['prompts'])
        assert not (set(c['checks']) & {p['frame_idx'] for p in c['prompts']})
        assert min(p['frame_idx'] for p in c['prompts'])==start
        source=PARENT/c['source_case'];fm=json.loads((source/'frames.json').read_text())
        for idx in range(start,end+1):
            f=source/'frames'/f'{idx:05d}.jpg';assert sha(f)==fm['jpeg_sha256'][idx]
            os.symlink(f,cache/f'{idx-start:05d}.jpg')
        hashes[c['source_case']]=dict(objects=sha(source/'recovery/objects.jsonl.gz'),
            complete=sha(source/'recovery/complete.json'),frames=sha(source/'frames.json'))
    write_json(out,dict(spec=spec,source=str(PARENT),source_config_sha256=sha(PARENT/'config.json'),
        source_hashes=hashes,script_sha256=sha(__file__),spec_sha256=sha(SPEC),sam2_sha256=sha(WEIGHTS)))
    print('PREPARED',len(spec['cases']),flush=True)


def load(root):
    cfg=json.loads((root/'config.json').read_text());assert sha(__file__)==cfg['script_sha256']
    assert sha(SPEC)==cfg['spec_sha256'] and sha(WEIGHTS)==cfg['sam2_sha256']
    assert sha(PARENT/'config.json')==cfg['source_config_sha256'];load_parent(PARENT)
    for sid,h in cfg['source_hashes'].items():
        assert sha(PARENT/sid/'recovery/objects.jsonl.gz')==h['objects']
        assert sha(PARENT/sid/'recovery/complete.json')==h['complete']
        assert sha(PARENT/sid/'frames.json')==h['frames']
    return cfg


def image_at(c,idx):
    image=cv2.imread(str(PARENT/c['source_case']/'frames'/f'{idx:05d}.jpg'))
    assert image is not None
    return image


def seeds(a):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=load(a.root);dest=a.root/'seeds';dest.mkdir(exist_ok=False);rows=[]
    predictor=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for c in cfg['spec']['cases']:
            for p in c['prompts']:
                image=image_at(c,p['frame_idx']);mask=np.zeros(image.shape[:2],bool);score=None
                if p['status']=='visible':
                    predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                    masks,scores,_=predictor.predict(box=np.array(p['box']),point_coords=np.array(p['points']),
                        point_labels=np.array(p['point_labels']),multimask_output=False)
                    mask=masks[0].astype(bool);score=float(scores[0]);assert mask.any()
                    for (x,y),label in zip(p['points'],p['point_labels']):
                        assert bool(mask[y,x])==bool(label),(c['case_id'],p['frame_idx'],x,y,label)
                r=dict(case_id=c['case_id'],object_id=c['object_id'],frame_idx=p['frame_idx'],
                    prompt=p,**record(mask),human_confirmed=False,sam_predicted_iou=score,
                    rgb_sha256=sha(PARENT/c['source_case']/'frames'/f'{p["frame_idx"]:05d}.jpg'))
                rows.append(r)
                rgb,_=labelled(image,[],f'{c["case_id"]} RGB {p["frame_idx"]}')
                view,_=labelled(image,[dict(r,class_name=str(c['object_id']),confidence=None)],p['status'])
                cv2.imwrite(str(dest/f'{c["case_id"]}_{p["frame_idx"]}.jpg'),np.concatenate([rgb,view],1))
    write_json(dest/'seeds.json',rows);print('SEEDS',len(rows),sha(dest/'seeds.json'),flush=True)


def run(a):
    from sam2.build_sam import build_sam2_video_predictor
    cfg=load(a.root);seedpath=a.root/'seeds/seeds.json';reviewpath=a.root/'seeds/review.json'
    review=json.loads(reviewpath.read_text());assert review['seed_sha256']==sha(seedpath)
    assert review['decision']=='use_as_assisted_seeds' and review['human_confirmed'] is False
    seeds=json.loads(seedpath.read_text())
    predictor=build_sam2_video_predictor(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    for ci,c in enumerate(cfg['spec']['cases']):
        if ci%a.shards!=a.shard:continue
        out=a.root/c['case_id']/'repair';out.mkdir(exist_ok=False)
        old=rows_at(PARENT/c['source_case']/'recovery/objects.jsonl.gz')
        events={r['frame_idx']:r for r in seeds if r['case_id']==c['case_id']}
        boundaries=sorted({c['start'],c['end']+1,*events,*range(c['start'],c['end']+1,30)})
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats();active=False;last=None;previous={}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),gzip.open(out/'objects.jsonl.gz','wt') as f:
            state=predictor.init_state(str(a.root/c['case_id']/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            for start,end in zip(boundaries[:-1],boundaries[1:]):
                if start in events:
                    if active:predictor.remove_object(state,c['object_id'],strict=True,need_output=False)
                    active=False;last=events[start];previous={}
                    if last['prompt']['status']=='visible':
                        predictor.add_new_mask(state,start-c['start'],c['object_id'],decode(last['mask']));active=True
                pred={}
                if active:
                    for idx,oids,logits in predictor.propagate_in_video(state,start_frame_idx=start-c['start'],max_frame_num_to_track=end-start-1):
                        assert oids==[c['object_id']]
                        cache=state['output_dict_per_obj'][0]
                        value=cache['cond_frame_outputs'].get(idx)
                        if value is None:value=cache['non_cond_frame_outputs'][idx]
                        pred[idx+c['start']]=((logits[0,0]>0).cpu().numpy(),float(value['object_score_logits'].float().sigmoid().item()))
                    assert set(pred)==set(range(start,end))
                for idx in range(start,end):
                    for oldrow in old[idx]:
                        row=copy.deepcopy(oldrow)
                        if row['object_id']==c['object_id']:
                            row.update(human_confirmed=False,mode='semantic_interval_repair',seed_frame=last['frame_idx'] if active else None,
                                provenance=dict(kind='assistant_semantic_interval',review_sha256=sha(reviewpath),
                                    seed_sha256=sha(seedpath),event_frame=last['frame_idx'],causal=True),
                                visibility='unknown',visible=None,mask=None,mask_area=None,bbox_xyxy=None,confidence=None,
                                mask_status='unknown',suspicious_flags=[])
                            if idx in pred:
                                m,prob=pred[idx];flags=diagnostics({c['object_id']:m},previous,None,Settings())[c['object_id']]
                                row.update(**record(m),confidence=prob,visibility='predicted_visible' if m.any() else 'unknown',suspicious_flags=flags)
                                previous={c['object_id']:m}
                            row['suspicious_flags'].append('assistant_repair_not_gt')
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                print('PROPAGATED',c['case_id'],end,c['end']+1,flush=True)
            del state
        write_json(out/'complete.json',dict(config_sha256=sha(a.root/'config.json'),seed_sha256=sha(seedpath),
            review_sha256=sha(reviewpath),objects_sha256=sha(out/'objects.jsonl.gz'),seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,semantic_acceptance=False))
        print('COMPLETE',c['case_id'],flush=True)


def finish(a):
    cfg=load(a.root);qa=a.root/'qa';qa.mkdir(exist_ok=False);report=[];merged={};keys=set()
    for c in cfg['spec']['cases']:
        source=PARENT/c['source_case'];original=rows_at(source/'recovery/objects.jsonl.gz')
        repaired=rows_at(a.root/c['case_id']/'repair/objects.jsonl.gz')
        meta=json.loads((a.root/c['case_id']/'repair/complete.json').read_text())
        assert sha(a.root/c['case_id']/'repair/objects.jsonl.gz')==meta['objects_sha256']
        assert meta['config_sha256']==sha(a.root/'config.json')
        assert set(repaired)==set(range(c['start'],c['end']+1))
        merged.setdefault(c['source_case'],copy.deepcopy(original));counts={'rows':0,'unchanged_other_objects':0,'unknown_target':0,'positive_target':0}
        fm=json.loads((source/'frames.json').read_text());dest=qa/c['case_id'];dest.mkdir();writer=Video(dest/'comparison.mp4',30)
        robotmeta=json.loads((source/'recovery/complete.json').read_text())['robot_parts_source']
        assert sha(robotmeta['path'])==robotmeta['sha256'];robot=rows_at(robotmeta['path'])
        for idx,rows in sorted(repaired.items()):
            before={r['object_id']:r for r in original[idx]};after={r['object_id']:r for r in rows}
            assert len(rows)==len(after) and set(before)==set(after)
            for oid,row in after.items():
                counts['rows']+=1;assert row['timestamp']==fm['timestamps'][idx]
                if oid!=c['object_id']:assert row==before[oid];counts['unchanged_other_objects']+=1;continue
                key=c['source_case'],idx,oid;assert key not in keys;keys.add(key)
                event=max(p['frame_idx'] for p in c['prompts'] if p['frame_idx']<=idx)
                assert row['provenance']['event_frame']==event and row['human_confirmed'] is False
                if row['mask'] is not None:
                    m=decode(row['mask']);g=record(m);assert m.shape==tuple(fm['shape'][:2])
                    assert row['bbox_xyxy']==g['bbox_xyxy'] and row['mask_area']==g['mask_area'];counts['positive_target']+=bool(m.any())
                else:assert row['visible'] is None and row['visibility']=='unknown';counts['unknown_target']+=1
                merged[c['source_case']][idx]=[row if r['object_id']==oid else r for r in merged[c['source_case']][idx]]
            image=image_at(c,idx);cells=[]
            for rr,title in [([],f'{c["case_id"]} RGB {idx}'),(original[idx],'Previous recovery'),(rows,'Local semantic repair'),(robot[idx],'Robot unchanged / not accepted')]:
                rgb,legend=labelled(image,rr,title);legend=np.pad(legend,((0,80-legend.shape[0]),(0,0),(0,0)))
                cells.append(np.concatenate([rgb,legend],0))
            canvas=np.concatenate([np.concatenate(cells[:2],1),np.concatenate(cells[2:],1)],0);writer.add(canvas)
            if idx in c['checks']:cv2.imwrite(str(dest/f'{idx:05d}.jpg'),canvas)
        writer.close();cap=cv2.VideoCapture(str(dest/'comparison.mp4'));n=0
        while True:
            ok,img=cap.read()
            if not ok:break
            assert img.shape==writer.shape;n+=1
        cap.release();assert n==c['end']-c['start']+1
        report.append(dict(case_id=c['case_id'],frames=n,video_sha256=sha(dest/'comparison.mp4'),robot_source=robotmeta,**counts))
        print('QA PASS',c['case_id'],n,flush=True)
    dest=a.root/'merged';dest.mkdir();mergereport=[]
    for sid,frames in merged.items():
        original=rows_at(PARENT/sid/'recovery/objects.jsonl.gz');changed=0;unchanged=0;path=dest/f'{sid}.jsonl.gz'
        with gzip.open(path,'wt') as f:
            for idx,rows in sorted(frames.items()):
                for row in rows:
                    old=next(r for r in original[idx] if r['object_id']==row['object_id'])
                    targeted=(sid,idx,row['object_id']) in keys
                    if not targeted:assert row==old;unchanged+=1
                    else:changed+=1
                    f.write(json.dumps(row,allow_nan=False)+'\n')
        mergereport.append(dict(source_case=sid,rows=changed+unchanged,modified_target_rows=changed,unchanged_rows=unchanged,sha256=sha(path)))
    write_json(a.root/'validation.json',dict(status='PASS',semantic_acceptance=False,cases=report,merged=mergereport))


def bundle_requests(requests,span=90):
    """Bundle delivery, without dropping triggers or marking any request resolved."""
    groups={}
    for i,r in enumerate(requests):groups.setdefault((r['case_id'],r['frame_idx']//span),[]).append((i,r))
    return [dict(case_id=key[0],start=min(r['frame_idx'] for _,r in members),end=max(r['frame_idx'] for _,r in members),
        status='pending_assistant_review',request_ids=[i for i,_ in members],
        members=[r for _,r in members],all_triggers=[t for _,r in members for t in r['triggers']]) for key,members in sorted(groups.items())]


def queue(a):
    requests=json.loads((PARENT/'recovery_review_queue.json').read_text());bundles=bundle_requests(requests)
    assert sorted(i for b in bundles for i in b['request_ids'])==list(range(len(requests)))
    assert sum(len(b['all_triggers']) for b in bundles)==sum(len(r['triggers']) for r in requests)
    cfg=load(a.root);reviewed=[]
    for c in cfg['spec']['cases']:
        for p in c['prompts']:reviewed.append(dict(source_case=c['source_case'],object_id=c['object_id'],frame_idx=p['frame_idx'],
            decision=p['status'],scope='exact_anchor_only',human_confirmed=False))
    write_json(a.root/'review_bundles.json',dict(source_sha256=sha(PARENT/'recovery_review_queue.json'),
        requests=len(requests),bundles=bundles,reviewed_anchors=reviewed,
        policy='Only delivery grouping; all original requests remain pending. No interval absence or semantic acceptance inferred.'))
    print('QUEUE',len(requests),'requests',len(bundles),'bundles',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['inspect','prepare','seeds','run','finish','queue'])
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20260918)
    dict(inspect=inspect,prepare=prepare,seeds=seeds,run=run,finish=finish,queue=queue)[a.phase](a)
