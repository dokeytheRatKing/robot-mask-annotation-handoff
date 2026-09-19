"""Versioned assistant seeds and bounded propagation for the fixed 64 clips."""
import argparse
import copy
import gzip
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from annotate import BASE, sha, write_json
from calibration64 import ROOT, load, read, propagate
from full_episode_reentry import rows_at
from masks import record
from seed_bank_transfer import MODEL, WEIGHTS
from seed_bank_transfer_qa import labelled

SPEC=BASE/'config/calibration64_repairs.json'
OUT=ROOT/'repairs'


def seeds(a):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=load(ROOT);spec=json.loads(SPEC.read_text());OUT.mkdir(exist_ok=True)
    predictor=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    results=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for number,e in enumerate(spec['events']):
            c=cfg['clips'][e['clip']];p=ROOT/'clips'/c['clip_id']/'rgb'/f'{e["frame"]-c["start"]:05d}.png'
            image=cv2.imread(str(p));predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            coords=e['positive']+e['negative'];labels=[1]*len(e['positive'])+[0]*len(e['negative'])
            masks,scores,_=predictor.predict(box=np.asarray(e['box']),point_coords=np.asarray(coords),point_labels=np.asarray(labels),multimask_output=False)
            r=dict(e,object_id=e['id'],frame_idx=e['frame'],class_name=str(e['id']),**record(masks[0].astype(bool)),
                confidence=None,sam_predicted_iou=float(scores[0]),human_confirmed=False,
                provenance=dict(kind='assistant_rgb_points_sam2',source_image_sha256=sha(p),spec_sha256=sha(SPEC)))
            results.append(r)
            rgb,_=labelled(image,[],f'{number}: C{e["clip"]:03d} frame{e["frame"]}');overlay,_=labelled(image,[r],f'Object{e["id"]}')
            cv2.imwrite(str(OUT/f'seed_{number:02d}.jpg'),np.concatenate([rgb,overlay],1))
    write_json(OUT/'seeds.json',results)
    print('SEEDS',len(results),flush=True)


def run(a):
    from sam2.build_sam import build_sam2_video_predictor
    cfg=load(ROOT);review=read(OUT,'seed_review.json');assert review['seed_sha256']==sha(OUT/'seeds.json')
    events=read(OUT,'seeds.json');accepted=set(review['accepted_event_indices'])
    predictor=build_sam2_video_predictor(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    for ci in sorted({e['clip'] for i,e in enumerate(events) if i in accepted}):
        if ci%a.shards!=a.shard:continue
        c=copy.deepcopy(cfg['clips'][ci]);path=ROOT/'clips'/c['clip_id'];original=rows_at(path/'draft/objects.jsonl.gz')
        selected=[e for i,e in enumerate(events) if i in accepted and e['clip']==ci]
        if ci==38:c['object_ids']=[0,4]
        byid=defaultdict(list)
        for e in selected:byid[e['object_id']].append(e)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred,anns=propagate(predictor,c,path,selected)
        dest=OUT/c['clip_id'];dest.mkdir(exist_ok=False)
        seedmap={(r['frame_idx'],r['object_id']):r for r in read(path,'accepted_seeds.json')}
        objects={o['object_id']:o for o in json.loads((BASE/'config/objects.json').read_text())}
        outrows={};modified=0
        with gzip.open(dest/'objects.jsonl.gz','wt',encoding='utf-8') as f:
            for idx in range(c['start'],c['end']+1):
                rows=[];existing={r['object_id']:r for r in original.get(idx,[])}
                for oid in c['object_ids']:
                    r=copy.deepcopy(existing.get(oid,dict(clip_id=c['clip_id'],episode_id=c['episode']['episode_id'],
                        task_id=c['task_id'],camera=c['camera'],object_id=oid,class_name=objects[oid]['class_name'],
                        frame_idx=idx,timestamp=c['timestamps'][idx-c['start']],image_width=c['shape'][1],image_height=c['shape'][0],
                        human_confirmed=False,training_eligible=False,suspicious_flags=[])))
                    if oid in byid:
                        # Exact accepted masks immutable, except explicit discrepant negatives -> UNKNOWN.
                        exact=(idx,oid) in seedmap
                        discrepancy=exact and (ci==47 and oid in [6,8,12] or ci==58 and oid==13)
                        if not exact or discrepancy:
                            e=anns[oid][idx];r.update(mask=None,mask_area=None,bbox_xyxy=None,visible=None,visibility='unknown',
                                confidence=None,mask_status='unknown',human_confirmed=False,training_eligible=False,
                                provenance=dict(kind='assistant_reseeded_sam2',seed_frame=e['frame_idx'],
                                    seed_sha256=sha(OUT/'seeds.json'),future_seed=e['frame_idx']>idx,labels_only=True,
                                    source_image_sha256=c['rgb_sha256'][idx-c['start']],parent_draft_sha256=sha(path/'draft/objects.jsonl.gz')))
                            modified+=1
                            if discrepancy:
                                r['suspicious_flags']=list(set(r.get('suspicious_flags',[])+['accepted_negative_discrepancy_unknown']))
                            elif oid in pred and idx in pred[oid] and pred[oid][idx][0].any():
                                m,score=pred[oid][idx];r.update(**record(m),confidence=score,visibility='predicted_visible')
                            if ci==38 and idx==208:
                                r.update(mask=None,mask_area=None,bbox_xyxy=None,visible=None,visibility='unknown',confidence=None,mask_status='unknown')
                                r['suspicious_flags']=['original_ignored_blur_preserved_unknown']
                    f.write(json.dumps(r,allow_nan=False)+'\n');rows.append(r)
                outrows[idx]=rows
        checks=sorted(set(np.linspace(c['start'],c['end'],7,dtype=int).tolist()+[e['frame_idx'] for e in selected]+[c['anchor_frame']]))
        tiles=[]
        for idx in checks:
            image=cv2.imread(str(path/'rgb'/f'{idx-c["start"]:05d}.png'));cells=[]
            for rows,title in [([],f'C{ci:03d} RGB {idx}'),(outrows[idx],'Repaired objects')]:
                view,legend=labelled(image,rows,title);legend=np.pad(legend,((0,max(0,80-legend.shape[0])),(0,0),(0,0)));cells.append(np.concatenate([view,legend],0))
            tiles.append(np.concatenate(cells,1))
        # Split sheets to keep native image readable in review tools.
        for k in range(0,len(tiles),4):cv2.imwrite(str(dest/f'qa_{k//4}.jpg'),np.concatenate(tiles[k:k+4],0))
        write_json(dest/'complete.json',dict(clip_id=c['clip_id'],modified_rows=modified,checks=checks,
            output_sha256=sha(dest/'objects.jsonl.gz'),seed_sha256=sha(OUT/'seeds.json'),script_sha256=sha(__file__)))
        print('REPAIRED',c['clip_id'],modified,flush=True)


def refine(a):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=load(ROOT);rows=read(OUT,'seeds.json');write_json(OUT/'seeds_initial.json',rows)
    predictor=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    from masks import decode
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in [9,16]:
            e=rows[i];c=cfg['clips'][e['clip']];im=cv2.imread(str(ROOT/'clips'/c['clip_id']/'rgb'/f'{e["frame"]-c["start"]:05d}.png'))
            predictor.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            if i==9:
                points=[[550,270],[592,274],[622,300],[475,335],[500,303],[564,352]];labels=[1,1,1,1,0,0]
                mm,ss,_=predictor.predict(point_coords=np.asarray(points),point_labels=np.asarray(labels),multimask_output=True)
                for k,m in enumerate(mm):
                    r=dict(e,**record(m.astype(bool)));rgb,_=labelled(im,[],f'holder candidate{k}');over,_=labelled(im,[r],'Mask')
                    cv2.imwrite(str(OUT/f'holder_candidate_{k}.jpg'),np.concatenate([rgb,over],1))
                np.savez_compressed(OUT/'holder_candidates.npz',masks=mm,scores=ss)
            else:
                mm,ss,_=predictor.predict(box=np.asarray([283,260,373,360]),point_coords=np.asarray([[329,306],[305,291],[394,285]]),
                    point_labels=np.asarray([1,1,0]),multimask_output=False)
                m=decode(e['mask'])|mm[0].astype(bool);e.update(**record(m));e['provenance']['refinement']='Union SAM avocado flesh + SAM pit; same object'
                rgb,_=labelled(im,[],'Avocado with pit');over,_=labelled(im,[e],'Refined mask')
                cv2.imwrite(str(OUT/'avocado_refined.jpg'),np.concatenate([rgb,over],1))
    write_json(OUT/'seeds.json',rows)


def accept(a):
    rows=read(OUT,'seeds.json');mm=np.load(OUT/'holder_candidates.npz')['masks']
    rows[9].update(**record(mm[1].astype(bool)))
    rows[9]['provenance']['refinement']='Assistant selected point-only multimask candidate1: holder excludes black gripper'
    write_json(OUT/'seeds.json',rows)
    write_json(OUT/'seed_review.json',dict(seed_sha256=sha(OUT/'seeds.json'),accepted_event_indices=list(range(20)),
        decision='assistant_reviewed_use_as_pseudo_label_seeds',human_confirmed=False,
        evidence='All20 RGB/mask seed pairs inspected; holder candidates0/1/2 and avocado pit refinement inspected; holder candidate1 chosen.',
        source_references_unchanged=True))


def late_seed(a):
    """Add explicitly inspected re-entry seeds to C050, preserving revision1."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    assert OUT!=ROOT/'repairs'
    OUT.mkdir(exist_ok=False)
    cfg=load(ROOT);c=cfg['clips'][50]
    events=[r for r in read(ROOT/'repairs','seeds.json') if r['clip']==50]
    p=ROOT/'clips'/c['clip_id']/'rgb'/'00080.png';im=cv2.imread(str(p))
    predictor=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    specs=[dict(clip=50,id=4,frame=395,box=[0,0,153,196],positive=[[90,47],[56,152],[126,91]],negative=[[27,93],[161,150]]),
        dict(clip=50,id=21,frame=395,box=[0,76,87,112],positive=[[29,93],[63,102]],negative=[[25,118],[88,86]])]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        for e in specs:
            mm,ss,_=predictor.predict(box=np.asarray(e['box']),point_coords=np.asarray(e['positive']+e['negative']),
                point_labels=np.asarray([1]*len(e['positive'])+[0]*len(e['negative'])),multimask_output=False)
            r=dict(e,object_id=e['id'],frame_idx=e['frame'],class_name=str(e['id']),**record(mm[0].astype(bool)),confidence=None,
                sam_predicted_iou=float(ss[0]),human_confirmed=False,provenance=dict(kind='assistant_rgb_reentry_seed',source_image_sha256=sha(p)))
            events.append(r);rgb,_=labelled(im,[],f'Reentry object{e["id"]}');over,_=labelled(im,[r],'Mask')
            cv2.imwrite(str(OUT/f'seed_{e["id"]}.jpg'),np.concatenate([rgb,over],1))
    write_json(OUT/'seeds.json',events)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['seeds','run','refine','accept','late_seed']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();OUT=a.output;torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20260919);globals()[a.phase](a)
