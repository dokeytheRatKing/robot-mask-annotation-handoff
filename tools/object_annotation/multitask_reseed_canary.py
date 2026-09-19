"""Bounded cross-task assistant-seeded SAM2 comparison; never edits corpus output."""
import argparse
from collections import defaultdict
import copy
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from annotate import BASE, PROJECT, sha, write_json
from data import frames, scan
from full_episode_reentry import rows_at
from masks import decode, record
from recovery import Settings, diagnostics
from seed_bank import accepted_rows
from seed_bank_transfer import MODEL, WEIGHTS, old_rows
from seed_bank_transfer_qa import Video, labelled

ROOT = PROJECT/'annotations/multitask_reseed_canary_20260919'
SPEC = BASE/'config/multitask_reseed_canary_prompts.json'
MODES = ('single_seed', 'multi_seed')


def prepare(a):
    a.root.mkdir(exist_ok=False)
    episodes = {e['episode_id']: e for e in scan(PROJECT/'datasets/lerobot/astribot_full_v21_rgb_h264')}
    groups = defaultdict(list)
    for r in accepted_rows():
        if r['source_set']=='accepted87' and r['task_id'] in (5,13,27) and r['object_id']<1000:
            assert not r.get('exclude_from_metrics')
            groups[(r['task_id'],r['episode_id'],r['camera'],r['frame_idx'])].append(r)
    cases=[]
    for (task, eid, cam, target), refs in sorted(groups.items()):
        ep=episodes[eid];start=target-90;end=min(ep['frames']-1,target+30)
        assert start>=0
        cid=f'T{task:02d}_{eid}_{cam}'
        out=a.root/cid;out.mkdir();(out/'frames').mkdir();(out/'anchors').mkdir()
        anchors=[start,target-45];timestamps={};hashes={};images={}
        for idx,ts,image in frames(ep,cam):
            if idx<start:continue
            if idx>end:break
            path=out/'frames'/f'{idx-start:05d}.jpg'
            assert cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,95])
            timestamps[str(idx)]=ts;hashes[str(idx)]=sha(path)
            if idx in anchors:
                path=out/'anchors'/f'{idx:05d}.png';assert cv2.imwrite(str(path),image)
                images[idx]=image
        assert len(timestamps)==end-start+1
        old,source=old_rows(ep,cam,start,end,'objects')
        _,robot=old_rows(ep,cam,start,end,'robot_parts')
        ids=sorted(r['object_id'] for r in refs)
        assert all(sorted(r['object_id'] for r in rs)==ids for rs in old.values())
        case=dict(case_id=cid,episode=ep,camera=cam,object_ids=ids,start=start,end=end,
            seed_frames=anchors,eval_frame=target,eval_sample=refs[0]['sample_id'],
            eval_source_hashes=sorted({r['source_manifest_sha256'] for r in refs}),
            original_objects=source,robot_parts=robot,timestamps=timestamps,jpeg_sha256=hashes,
            shape=list(image.shape))
        cases.append(case)
        tiles=[]
        for idx,im in images.items():
            tile=cv2.resize(im,(640,360));tile=np.pad(tile,((30,0),(0,0),(0,0)))
            cv2.putText(tile,f'{cid} frame {idx}; coordinates 640x360',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            tiles.append(tile)
        assert cv2.imwrite(str(out/'anchors/contact.jpg'),np.concatenate(tiles,0))
    write_json(a.root/'config.json',dict(schema='astribot.multitask_reseed_canary.v1',cases=cases,
        tasks=[5,13,27],sam2_sha256=sha(WEIGHTS),sam2_config=MODEL,random_seed=20260919,
        design='Same first assistant seed; multi_seed resets memory and reseeds at eval-45. Single seed at eval-90.',
        reference_policy='Existing accepted87 biased development labels; never used as seeds. No claims of unseen benchmark accuracy.',
        scope='Object-only bounded windows, unchanged frozen robot parts for QA. No full corpus promotion.',
        unknown_policy='Absent/uncertain anchors and empty predictions abstain UNKNOWN; not confirmed interval negatives.',
        coordinate_policy='Prompt coordinates on 640x360 resized original anchor RGB; SAM2 inference at native resolution.',
        frozen_helper_hashes={n:sha(BASE/n) for n in ['masks.py','data.py','recovery.py','seed_bank_transfer.py','seed_bank_transfer_qa.py']}))
    print('PREPARED',len(cases),sum(c['end']-c['start']+1 for c in cases),flush=True)


def load(root):
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    assert sha(WEIGHTS)==cfg['sam2_sha256']
    for n,h in cfg['frozen_helper_hashes'].items():assert sha(BASE/n)==h,n
    return cfg


def read_image(root,c,idx):
    return cv2.imread(str(root/c['case_id']/'frames'/f'{idx-c["start"]:05d}.jpg'))


def seeds(a):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=load(a.root);spec=json.loads(SPEC.read_text(encoding='utf-8'))
    dest=a.root/'seeds';dest.mkdir(exist_ok=False);rows=[]
    predictor=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for c in cfg['cases']:
            prompts=spec[c['case_id']]
            assert sorted(map(int,prompts))==c['seed_frames']
            tiles=[]
            for idx in c['seed_frames']:
                ip=a.root/c['case_id']/'anchors'/f'{idx:05d}.png';image=cv2.imread(str(ip))
                h,w=image.shape[:2];predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                assert sorted(map(int,prompts[str(idx)]))==c['object_ids']
                rr=[]
                for oid in c['object_ids']:
                    p=prompts[str(idx)][str(oid)];m=np.zeros((h,w),bool);score=None
                    if p['status']=='visible':
                        parts=[]
                        for comp in p.get('components',[p]):
                            points=np.asarray(comp['points'],dtype=float)*[w/640,h/360]
                            box=np.asarray(comp['box'],dtype=float)*[w/640,h/360,w/640,h/360]
                            masks,scores,_=predictor.predict(box=box,point_coords=points,
                                point_labels=np.asarray(comp['point_labels']),multimask_output=False)
                            parts.append(masks[0].astype(bool));score=float(scores[0])
                        m=np.logical_or.reduce(parts);assert m.any()
                    r=dict(case_id=c['case_id'],frame_idx=idx,object_id=oid,prompt=p,**record(m),
                        class_name=str(oid),confidence=None,human_confirmed=False,
                        sam_predicted_iou=score,anchor_sha256=sha(ip))
                    rows.append(r);rr.append(r)
                rgb,_=labelled(image,[],f'{c["case_id"]} RGB {idx}')
                overlay,legend=labelled(image,rr,'Assistant point/box SAM2 seeds')
                tiles.append(np.concatenate([rgb,overlay],1))
            cv2.imwrite(str(dest/f'{c["case_id"]}.jpg'),np.concatenate(tiles,0))
    write_json(dest/'seeds.json',rows)
    write_json(dest/'manifest.json',dict(seed_sha256=sha(dest/'seeds.json'),prompt_sha256=sha(SPEC),
        config_sha256=sha(a.root/'config.json'),script_sha256=sha(__file__),human_confirmed=False))
    print('SEEDS',len(rows),flush=True)


def propagate(predictor,root,c,events,start,end):
    """All object decisions are explicit. Never project an absent anchor into a GT negative."""
    positives=[e for e in events if e['prompt']['status']=='visible']
    predictions={}
    if not positives:return predictions
    state=predictor.init_state(str(root/c['case_id']/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
    for e in positives:predictor.add_new_mask(state,start-c['start'],e['object_id'],decode(e['mask']))
    for local,oids,logits in predictor.propagate_in_video(state,start_frame_idx=start-c['start'],max_frame_num_to_track=end-start):
        assert set(oids)=={e['object_id'] for e in positives}
        predictions[local+c['start']]={}
        for pos,oid in enumerate(oids):
            cache=state['output_dict_per_obj'][state['obj_id_to_idx'][oid]]
            val=cache['cond_frame_outputs'].get(local)
            if val is None:val=cache['non_cond_frame_outputs'][local]
            predictions[local+c['start']][oid]=((logits[pos,0]>0).cpu().numpy(),float(val['object_score_logits'].float().sigmoid().item()))
    assert set(predictions)==set(range(start,end+1))
    del state
    return predictions


def run(a):
    from sam2.build_sam import build_sam2_video_predictor
    cfg=load(a.root);sp=a.root/'seeds/seeds.json';mp=a.root/'seeds/manifest.json'
    meta=json.loads(mp.read_text());review=json.loads((a.root/'seeds/review.json').read_text())
    assert meta['seed_sha256']==sha(sp)==review['seed_sha256']
    assert review['decision']=='use_as_assisted_seeds' and review['human_confirmed'] is False
    assert meta['prompt_sha256']==sha(SPEC) and meta['script_sha256']==sha(__file__)
    seedrows=json.loads(sp.read_text());predictor=build_sam2_video_predictor(MODEL,str(WEIGHTS),device='cuda',
        apply_postprocessing=False,hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    for ci,c in enumerate(cfg['cases']):
        if ci%a.shards!=a.shard:continue
        out=a.root/c['case_id']/'predictions';out.mkdir(exist_ok=False)
        for idx,h in c['jpeg_sha256'].items():assert sha(a.root/c['case_id']/'frames'/f'{int(idx)-c["start"]:05d}.jpg')==h
        source=c['original_objects'];assert sha(source['path'])==source['sha256'];old=rows_at(source['path'])
        ev={idx:[r for r in seedrows if r['case_id']==c['case_id'] and r['frame_idx']==idx] for idx in c['seed_frames']}
        s0,s1=c['seed_frames'];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            single=propagate(predictor,a.root,c,ev[s0],s0,c['end'])
            later=propagate(predictor,a.root,c,ev[s1],s1,c['end'])
        timing=time.perf_counter()-started
        for mode in MODES:
            previous={}
            with gzip.open(out/f'{mode}.jsonl.gz','wt',encoding='utf-8') as f:
                for idx in range(c['start'],c['end']+1):
                    event=s0 if mode=='single_seed' or idx<s1 else s1
                    preds=(single if event==s0 else later).get(idx,{})
                    if idx==event:previous={}
                    masks={oid:m for oid,(m,_) in preds.items()}
                    flags=diagnostics(masks,previous,None,Settings()) if masks else {}
                    for oldrow in old[idx]:
                        r=copy.deepcopy(oldrow);oid=r['object_id'];decision=next(e for e in ev[event] if e['object_id']==oid)
                        r.update(mask=None,bbox_xyxy=None,mask_area=None,visible=None,visibility='unknown',confidence=None,
                            mask_status='unknown',human_confirmed=False,mode=mode,seed_frame=event,
                            provenance=dict(kind='assistant_reseed_canary',seed_sha256=sha(sp),event_frame=event,
                                event_status=decision['prompt']['status'],causal=True,source_sha256=source['sha256']),
                            suspicious_flags=['assistant_prediction_not_gt'])
                        if oid in preds and preds[oid][0].any():
                            m,prob=preds[oid];r.update(**record(m),confidence=prob,visibility='predicted_visible')
                        r['suspicious_flags']+=flags.get(oid,[])
                        f.write(json.dumps(r,allow_nan=False)+'\n')
                    previous=masks
        write_json(out/'complete.json',dict(config_sha256=sha(a.root/'config.json'),seed_sha256=sha(sp),
            script_sha256=sha(__file__),seconds=timing,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            output_sha256={m:sha(out/f'{m}.jsonl.gz') for m in MODES},semantic_acceptance=False))
        print('COMPLETE',c['case_id'],round(timing,2),flush=True)


def summarize(rows):
    result={}
    for mode in ('frozen_production',*MODES):
        rr=[r for r in rows if r['mode']==mode];vis=[r for r in rr if r['gt_visible']];absent=[r for r in rr if not r['gt_visible']]
        result[mode]=dict(labels=len(rr),visible=len(vis),absent=len(absent),
            mean_iou_visible=float(np.mean([r['iou'] for r in vis])) if vis else None,
            mean_dice_visible=float(np.mean([r['dice'] for r in vis])) if vis else None,
            iou_ge_05=sum(r['iou']>=.5 for r in vis),visible_misses=sum(not r['pred_positive'] for r in vis),
            absent_false_positives=sum(r['pred_positive'] for r in absent),
            unknown=sum(r['unknown'] for r in rr),
            positive_presence_accuracy=sum(r['pred_positive']==r['gt_visible'] for r in rr)/len(rr))
    return result


def finish(a):
    cfg=load(a.root);refs=accepted_rows();metrics=[];counts=[];qa=a.root/'qa';qa.mkdir(exist_ok=False)
    for c in cfg['cases']:
        cid=c['case_id'];src=a.root/cid/'predictions';meta=json.loads((src/'complete.json').read_text())
        assert meta['config_sha256']==sha(a.root/'config.json') and meta['seed_sha256']==sha(a.root/'seeds/seeds.json')
        oldsource=c['original_objects'];robotsource=c['robot_parts']
        assert sha(oldsource['path'])==oldsource['sha256'] and sha(robotsource['path'])==robotsource['sha256']
        layers={'frozen_production':rows_at(oldsource['path'])};robot=rows_at(robotsource['path'])
        gt=[r for r in refs if r['sample_id']==c['eval_sample'] and r['object_id']<1000]
        assert sorted(r['object_id'] for r in gt)==c['object_ids']
        assert sorted({r['source_manifest_sha256'] for r in gt})==c['eval_source_hashes']
        for mode in MODES:
            p=src/f'{mode}.jsonl.gz';assert sha(p)==meta['output_sha256'][mode];layers[mode]=rows_at(p)
            assert set(layers[mode])==set(range(c['start'],c['end']+1))
            for idx,rr in layers[mode].items():
                assert sorted(r['object_id'] for r in rr)==c['object_ids']
                event=c['start'] if mode=='single_seed' or idx<c['seed_frames'][1] else c['seed_frames'][1]
                for r in rr:
                    assert r['timestamp']==c['timestamps'][str(idx)] and r['seed_frame']==event
                    assert r['human_confirmed'] is False and r['provenance']['causal']
                    if r['mask'] is not None:
                        binary=decode(r['mask']);geom=record(binary)
                        assert binary.shape==tuple(c['shape'][:2]) and geom['bbox_xyxy']==r['bbox_xyxy'] and geom['mask_area']==r['mask_area']
                    else:assert r['visible'] is None and r['visibility']=='unknown'
                if idx<c['seed_frames'][1] and mode=='multi_seed':
                    for x,y in zip(rr,layers['single_seed'][idx]):
                        xx=dict(x);yy=dict(y);xx.pop('mode');yy.pop('mode');assert xx==yy
        for mode,rr in layers.items():
            pred={r['object_id']:r for r in rr[c['eval_frame']]}
            for g in gt:
                r=pred[g['object_id']];t=decode(g['mask']);p=decode(r['mask']) if r['mask'] else np.zeros_like(t)
                inter=int((p&t).sum());union=int((p|t).sum());den=int(p.sum()+t.sum())
                metrics.append(dict(case_id=cid,camera=c['camera'],task_id=c['episode']['task_id'],object_id=g['object_id'],
                    mode=mode,eval_frame=c['eval_frame'],gt_visible=bool(g['visible']),pred_positive=bool(p.any()),
                    unknown=r.get('visible') is None,iou=inter/union if union else None,dice=2*inter/den if den else None,
                    gt_pixels=int(t.sum()),pred_pixels=int(p.sum()),seed_gap=c['eval_frame']-r['seed_frame'] if mode in MODES else None))
        out=qa/cid;out.mkdir();writer=Video(out/'comparison.mp4',c['episode']['fps'])
        checks=sorted(set([c['start']+15,c['seed_frames'][1]+15,c['eval_frame'],c['end']]))
        for idx in range(c['start'],c['end']+1):
            image=read_image(a.root,c,idx);cells=[]
            items=[([],f'{cid} RGB {idx}'),(layers['frozen_production'][idx],'Frozen production'),
                (robot[idx],'Robot unchanged, not accepted'),(layers['single_seed'][idx],'Single assisted seed'),
                (layers['multi_seed'][idx],'Two assisted keyframes'),(gt if idx==c['eval_frame'] else [],'Accepted GT (score frame only)')]
            for rr,title in items:
                rgb,legend=labelled(image,rr,title);legend=np.pad(legend,((0,80-legend.shape[0]),(0,0),(0,0)))
                cells.append(np.concatenate([rgb,legend],0))
            canvas=np.concatenate([np.concatenate(cells[:3],1),np.concatenate(cells[3:],1)],0);writer.add(canvas)
            if idx in checks:assert cv2.imwrite(str(out/f'{idx:05d}.jpg'),canvas)
        writer.close();cap=cv2.VideoCapture(str(out/'comparison.mp4'));n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==writer.shape;n+=1
        cap.release();assert n==c['end']-c['start']+1
        counts.append(dict(case_id=cid,frames=n,rows_per_mode=n*len(c['object_ids']),
            qa_sha256=sha(out/'comparison.mp4'),seconds=meta['seconds'],peak_allocated_gib=meta['peak_allocated_gib']))
        print('VALIDATED',cid,n,flush=True)
    write_json(a.root/'metrics.json',dict(per_label=metrics,overall=summarize(metrics),
        per_camera={cam:summarize([r for r in metrics if r['camera']==cam]) for cam in ['head','left_wrist','right_wrist']},
        per_task={str(t):summarize([r for r in metrics if r['task_id']==t]) for t in [5,13,27]},
        caveat='Biased development GT. Positive-presence accuracy counts UNKNOWN as non-positive, not verified negative visibility.'))
    write_json(a.root/'validation.json',dict(status='PASS',semantic_acceptance=False,cases=counts,
        total_frames=sum(c['frames'] for c in counts),rows_per_mode=sum(c['rows_per_mode'] for c in counts),
        gt_labels_per_mode=len(metrics)//3,new_manual_frames=0))
    print(json.dumps(summarize(metrics),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','seeds','run','finish'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20260919)
    globals()[a.phase](a)
