"""Bounded frozen-bank transfer and downstream SAM2 propagation comparison."""
import argparse
from collections import Counter, defaultdict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from annotate import BASE, PROJECT, QAVideo, sha, write_json
from data import frames, scan
from masks import decode, record
from recovery import Settings, diagnostics
from repair_confirmed_windows import painted
from seed_bank import CAMERAS, SeedBank, accepted_rows, crops, rank_score
from seed_bank_pilot import metrics

PRODUCTION=PROJECT/'annotations/segmentation_full_robot_20260917'
MODES=('detector','bank_rank','bank_gate')
MODEL='configs/sam2.1/sam2.1_hiera_b+.yaml'
WEIGHTS=PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt'


def choose(options,mode,eligible_first=False):
    if eligible_first:options=[c for c in options if c['detector_score']>=.30]
    ranked=sorted(options,key=lambda c:(-c['scores']['detector' if mode=='detector' else 'masked'],c['candidate_id']))
    if not ranked:return None,'no_candidate'
    c=ranked[0]
    if c['detector_score']<.30:return None,'detector_below_threshold'
    if mode=='bank_gate':
        r=c['retrieval']
        if r['similarity'] is None or r['margin'] is None:return None,'no_reference'
        if r['similarity']<.50:return None,'low_similarity'
        if r['margin']<.05:return None,'low_identity_margin'
    return c,'selected'


def prepare(a):
    a.output.mkdir(parents=True,exist_ok=False)
    bank=SeedBank(a.bank); accepted=accepted_rows()
    bank_hash=sha(a.bank/'bank.json')
    write_json(a.output/'bank_acceptance.json',dict(bank_sha256=bank_hash,
        review_bundle='deliverables/astribot_identity_seed_bank_review_20260918.zip',
        acceptance='User stated: 审核完毕，全部通过。',scope='Approve curated active/excluded decisions unchanged',
        new_manual_frames=0,source='conversation 2026-09-18',status='user_accepted'))
    episodes=scan(PROJECT/'datasets/lerobot/astribot_full_v21_rgb_h264')
    byid={e['episode_id']:e for e in episodes}
    excluded={r['episode_id'] for r in accepted}|{r['episode_id'] for r in bank.entries}
    rng=random.Random(20260918); cases=[]
    for task,side in [(1,'left'),(9,'right'),(24,'left'),(24,'right')]:
        eligible=sorted([e for e in episodes if e['task_id']==task and e['side']==side and
                         e['episode_id'] not in excluded and e['frames']>=121],key=lambda e:e['episode_id'])
        ep=rng.choice(eligible); start=max(0,(ep['frames']-121)//2)
        for cam in CAMERAS:
            cases.append(dict(case_id=f'new_T{task}_{side}_{cam}',kind='unseen_episode',episode=ep,
                camera=cam,start=start,end=start+120,seed_frames=[start,start+40,start+80],eval_frame=None))
    groups=defaultdict(list)
    for r in accepted:
        if r['source_set']=='accepted87' and r['task_id'] in (1,9,24) and r['object_id']<1000 and not r.get('exclude_from_metrics'):
            groups[r['sample_id']].append(r)
    for sid,labels in sorted(groups.items()):
        r=labels[0];ep=byid[r['episode_id']];target=r['frame_idx']
        assert target>=15 and target+15<ep['frames']
        cases.append(dict(case_id='score_'+sid,kind='accepted_nonseed',episode=ep,camera=r['camera'],
            start=target-15,end=target+15,seed_frames=[target-15],eval_frame=target,
            eval_sample=sid,eval_source_hashes=sorted({x['source_manifest_sha256'] for x in labels})))
    if a.scoring_only:cases=[c for c in cases if c['kind']=='accepted_nonseed']
    mapping=json.loads((PRODUCTION/'config/task_objects.json').read_text())
    for c in cases:c['object_ids']=mapping[str(c['episode']['task_id'])]
    cfg=dict(schema='astribot.seed_bank_transfer.v1',bank=str(a.bank.resolve()),bank_sha256=bank_hash,
        acceptance_sha256=sha(a.output/'bank_acceptance.json'),script_sha256=sha(__file__),
        seed_bank_code_sha256=sha(BASE/'seed_bank.py'),encoder_code_sha256=sha(BASE/'identity_guard.py'),
        detector_code_sha256=sha(BASE/'detector.py'),sam2_sha256=sha(WEIGHTS),
        dino_sha256=sha(PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'),
        encoder_sha256=sha(PROJECT/'models/dinov2/dinov2_vits14_pretrain.pth'),
        excluded_source_episodes=sorted(excluded),cases=cases,modes=list(MODES),random_seed=20260918,
        eligible_first=a.eligible_first,
        candidate_parent=str(a.candidate_parent.resolve()) if a.candidate_parent else None,
        policy='Frozen masked ranking; .30 detector, .50 cosine, .05 margin. No threshold tuning.',
        unknown='No seed is UNKNOWN, not a confirmed negative. Scoring an abstention as empty penalizes visible misses.',
        propagation='Independent forward SAM2 segments, deterministic periodic re-detection at 40 frames on new clips.',
        limitations=['Short-window recovery comparison, not full episode reentry or corpus accuracy.',
                    'GT scoring frames are not seed frames; entire query episode excluded from bank retrieval.',
                    'Accepted scoring set is a biased development set. New episode review is qualitative.',
                    'Only object propagation varies; frozen robot layers stay QA-only, never subtracted.'])
    write_json(a.output/'run_config.json',cfg)
    print(json.dumps([dict(case_id=c['case_id'],episode=c['episode']['episode_id'],start=c['start'],end=c['end']) for c in cases],indent=2))


def candidates(image,ids,ep,cam,anchor,detector,segmenter,encoder,bank):
    result=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        segmenter.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    for oid in ids:
        ds,_=detector.detect(image,[oid],floor=.10,nms_iou=.5)
        for d in sorted(ds,key=lambda r:-r['confidence'])[:5]:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                masks,scores,_=segmenter.predict(box=np.array(d['bbox_xyxy']),multimask_output=False)
            m=masks[0].astype(bool)
            if not m.any():continue
            rgb,cutout,_,_=crops(image,m)
            result.append(dict(candidate_id=f'{anchor}_{oid}_{len(result)}',object_id=oid,
                detector_score=d['confidence'],sam_predicted_iou=float(scores[0]),binary=m,rgb=rgb,cutout=cutout))
    qr=encoder.embed([c['rgb'] for c in result]).cpu().numpy()
    qm=encoder.embed([c['cutout'] for c in result]).cpu().numpy()
    for i,c in enumerate(result):
        c['retrieval']=bank.query(qr[i],qm[i],c['object_id'],cam,exclude_episode=ep,mode='masked')
        c['scores']=dict(detector=c['detector_score'],masked=rank_score(c['detector_score'],c['retrieval']))
    return result


def old_rows(ep,cam,start,end,layer):
    directory=PRODUCTION/f'task_{ep["task_id"]:02d}'/ep['episode_id']/cam
    paths=list(directory.glob(f'attempt_*/{layer}.jsonl.gz'))
    assert len(paths)==1,(directory,paths)
    result=defaultdict(list)
    with gzip.open(paths[0],'rt') as f:
        for line in f:
            r=json.loads(line)
            if r['frame_idx']>end:break
            if r['frame_idx']>=start:result[r['frame_idx']].append(r)
    assert set(result)==set(range(start,end+1))
    return result,dict(path=str(paths[0]),sha256=sha(paths[0]))


def panel(image,rows,label):
    # Legacy robot-union records use string IDs; normalize only the drawing copy.
    drawing=[dict(r,object_id=(r['object_id'] if isinstance(r['object_id'],int) else
        int.from_bytes(hashlib.sha256(str(r['object_id']).encode()).digest()[:4],'little')))
        for r in rows]
    out=cv2.resize(painted(image,drawing),(426,240))
    out=np.pad(out,((26,0),(0,0),(0,0)))
    cv2.putText(out,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    return out


def run(a):
    cfg=json.loads((a.output/'run_config.json').read_text());assert cfg['script_sha256']==sha(__file__)
    bank=SeedBank(Path(cfg['bank']));assert cfg['bank_sha256']==sha(Path(cfg['bank'])/'bank.json')
    torch.set_num_threads(3);cv2.setNumThreads(1);torch.manual_seed(20260918)
    from detector import Detector
    from identity_guard import Appearance
    from sam2.build_sam import build_sam2,build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    objects={o['object_id']:o for o in json.loads((BASE/'config/objects.json').read_text())}
    parent=Path(cfg['candidate_parent']) if cfg.get('candidate_parent') else None
    if parent:
        parent_cfg=json.loads((parent/'run_config.json').read_text())
        assert parent_cfg['bank_sha256']==cfg['bank_sha256']
        parent_cases={c['case_id']:c for c in parent_cfg['cases']}
    else:
        detector=Detector(str(PROJECT/'third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
            str(PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'),str(PROJECT/'models/groundingdino/bert-base-uncased'),objects)
        segmenter=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
        encoder=Appearance()
    predictor=build_sam2_video_predictor(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    for i,c in enumerate(cfg['cases']):
        if i%a.shards!=a.shard:continue
        directory=a.output/c['case_id'];directory.mkdir(exist_ok=False)
        start=time.perf_counter();ep=c['episode'];cam=c['camera'];ids=c['object_ids']
        cache=directory/'frames';cache.mkdir();original={};timestamps={}
        for idx,ts,image in frames(ep,cam):
            if idx<c['start']:continue
            if idx>c['end']:break
            original[idx]=image;timestamps[idx]=ts
            assert cv2.imwrite(str(cache/f'{idx-c["start"]:05d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,98])
        assert len(original)==c['end']-c['start']+1
        old,oldsrc=old_rows(ep,cam,c['start'],c['end'],'objects')
        robot,robotsrc=old_rows(ep,cam,c['start'],c['end'],'robot')
        selections={};output={m:{} for m in MODES};selectionlog=[]
        if parent:
            assert parent_cases[c['case_id']]==c
            parent_meta=json.loads((parent/c['case_id']/'complete.json').read_text())
            parent_path=parent/c['case_id']/'candidates.jsonl'
            assert sha(parent_path)==parent_meta['candidates_sha256']
            cached=[json.loads(s) for s in parent_path.read_text().splitlines()]
            write_json(directory/'candidate_source.json',dict(path=str(parent_path),sha256=sha(parent_path)))
        torch.cuda.reset_peak_memory_stats()
        with (directory/'candidates.jsonl').open('x') as f:
            for anchor in c['seed_frames']:
                pool=([dict(r,binary=decode(r['mask'])) for r in cached if r['frame_idx']==anchor] if parent else
                      candidates(original[anchor],ids,ep['episode_id'],cam,anchor,detector,segmenter,encoder,bank))
                for proposal in pool:
                    payload={k:v for k,v in proposal.items() if k not in ('binary','rgb','cutout')}
                    payload.update(frame_idx=anchor,**record(proposal['binary']))
                    f.write(json.dumps(payload,allow_nan=False)+'\n')
                for mode in MODES:
                    for oid in ids:
                        options=[x for x in pool if x['object_id']==oid]
                        chosen,reason=choose(options,mode,cfg.get('eligible_first',False))
                        selections[mode,anchor,oid]=chosen
                        selectionlog.append(dict(mode=mode,frame_idx=anchor,object_id=oid,reason=reason,
                            candidate_id=chosen['candidate_id'] if chosen else None))
                print(f'{c["case_id"]} candidates frame {anchor}',flush=True)
        write_json(directory/'selections.json',selectionlog)
        times={};flags=Counter()
        for mode in MODES:
            t=time.perf_counter();previous={}
            with gzip.open(directory/f'{mode}.jsonl.gz','wt') as f:
                for k,anchor in enumerate(c['seed_frames']):
                    end=c['seed_frames'][k+1]-1 if k+1<len(c['seed_frames']) else c['end']
                    seeds={oid:selections[mode,anchor,oid] for oid in ids if selections[mode,anchor,oid] is not None}
                    predictions={}
                    if seeds:
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            state=predictor.init_state(str(cache),offload_video_to_cpu=True,offload_state_to_cpu=True)
                            for oid,seed in seeds.items():predictor.add_new_mask(state,anchor-c['start'],oid,seed['binary'])
                            for local,oids,logits in predictor.propagate_in_video(state,start_frame_idx=anchor-c['start'],max_frame_num_to_track=end-anchor):
                                actual=local+c['start']
                                if actual>end:break
                                predictions[actual]={}
                                for j,(oid,m) in enumerate(zip(oids,(logits[:,0]>0).cpu().numpy())):
                                    cached=state['output_dict_per_obj'][j]
                                    value=cached['cond_frame_outputs'].get(local)
                                    if value is None:value=cached['non_cond_frame_outputs'][local]
                                    prob=float(value['object_score_logits'].float().sigmoid().reshape(-1)[0])
                                    predictions[actual][oid]=(m,prob)
                            del state
                        assert set(predictions)==set(range(anchor,end+1))
                    for idx in range(anchor,end+1):
                        rows=[];binary_byid={oid:pair[0] for oid,pair in predictions.get(idx,{}).items()}
                        robot_mask=decode(robot[idx][0]['mask'])
                        warnings=diagnostics(binary_byid,previous,robot_mask,Settings())
                        for oid in ids:
                            common=dict(episode_id=ep['episode_id'],task_id=ep['task_id'],camera=cam,frame_idx=idx,
                                timestamp=timestamps[idx],object_id=oid,class_name=objects[oid]['class_name'],
                                human_confirmed=False,mode=mode,seed_frame=anchor,
                                image_width=original[idx].shape[1],image_height=original[idx].shape[0])
                            seed=selections[mode,anchor,oid]
                            if seed is None:
                                r=dict(common,mask=None,bbox_xyxy=None,mask_area=None,visible=None,confidence=None,
                                    visibility='unknown',suspicious_flags=['unseeded_unknown'])
                            else:
                                m,prob=predictions[idx][oid]
                                r=dict(common,**record(m),confidence=prob,visibility='predicted_visible' if m.any() else 'unknown',
                                    suspicious_flags=warnings.get(oid,[]),provenance=dict(candidate_id=seed['candidate_id'],
                                        detector_score=seed['detector_score'],bank=seed['retrieval'],seed_frame=anchor,
                                        kind='automatic_reseed_not_gt',bank_sha256=cfg['bank_sha256']))
                            rows.append(r);f.write(json.dumps(r,allow_nan=False)+'\n')
                            for flag in r['suspicious_flags']:flags[f'{mode}/{flag}']+=1
                        output[mode][idx]=rows;previous=binary_byid
                    del predictions
                torch.cuda.synchronize();times[mode]=time.perf_counter()-t
            gc.collect()
            print(f'{c["case_id"]} {mode} done {times[mode]:.1f}s',flush=True)
        contacts=set(np.linspace(c['start'],c['end'],5,dtype=int).tolist())|set(c['seed_frames'])
        if c['eval_frame'] is not None:contacts.add(c['eval_frame'])
        # Legacy QAVideo has fixed geometry; explicit resize keeps its contract.
        writer=QAVideo(directory/'comparison.mp4',ep['fps']);sheets=[]
        for idx,image in original.items():
            panels=[panel(image,[],f'{c["case_id"]} {idx} RGB'),panel(image,old[idx],'Frozen production'),
                    *[panel(image,output[m][idx],m) for m in MODES],panel(image,robot[idx],'Robot QA only, unchanged')]
            canvas=np.concatenate([np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)],0)
            writer.add(cv2.resize(canvas,(1280,392)))
            if idx in contacts:
                cv2.imwrite(str(directory/f'qa_{idx}.jpg'),canvas);sheets.append(canvas)
        writer.close();cv2.imwrite(str(directory/'overview.jpg'),np.concatenate(sheets,0))
        write_json(directory/'complete.json',dict(status='COMPLETE',case=c,seconds=time.perf_counter()-start,
            propagation_seconds=times,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            flags=dict(flags),old_source=oldsrc,robot_source=robotsrc,
            frame_count=len(original),output_sha256={m:sha(directory/f'{m}.jsonl.gz') for m in MODES},
            candidates_sha256=sha(directory/'candidates.jsonl'),selections_sha256=sha(directory/'selections.json')))
        print(f'COMPLETE {c["case_id"]} {time.perf_counter()-start:.1f}s',flush=True)
        del output,original,old,robot,selections
        gc.collect()


def score(a):
    cfg=json.loads((a.output/'run_config.json').read_text());accepted=accepted_rows();results=[];done=[]
    for case in cfg['cases']:
        directory=a.output/case['case_id'];complete=json.loads((directory/'complete.json').read_text());done.append(complete)
        if case['eval_frame'] is None:continue
        idx=case['eval_frame'];assert idx not in case['seed_frames']
        labels=[r for r in accepted if r['sample_id']==case['eval_sample'] and r['object_id']<1000 and not r.get('exclude_from_metrics')]
        predictions={}
        for mode in MODES:
            assert sha(directory/f'{mode}.jsonl.gz')==complete['output_sha256'][mode]
            with gzip.open(directory/f'{mode}.jsonl.gz','rt') as f:
                predictions[mode]=[r for r in map(json.loads,f) if r['frame_idx']==idx]
        old,_=old_rows(case['episode'],case['camera'],idx,idx,'objects');predictions['frozen_production']=old[idx]
        for label in labels:
            gt=decode(label['mask']);recorded=original_frame(case,idx)
            assert np.array_equal(recorded,cv2.imread(label['rgb_path'])),'Scoring RGB alignment failed'
            for mode,rows in predictions.items():
                rs=[r for r in rows if r['object_id']==label['object_id'] and r.get('mask') is not None]
                pred=np.logical_or.reduce([decode(r['mask']) for r in rs]) if rs else np.zeros_like(gt)
                results.append(dict(case_id=case['case_id'],camera=case['camera'],object_id=label['object_id'],
                    frame_idx=idx,mode=mode,gt_visible=bool(gt.any()),pred_visible=bool(pred.any()),
                    abstained=not rs,**metrics(pred,gt)))
    summary=[]
    for camera in ('all',*CAMERAS):
        for mode in (*MODES,'frozen_production'):
            rows=[r for r in results if r['mode']==mode and (camera=='all' or r['camera']==camera)]
            pos=[r for r in rows if r['gt_visible']];neg=[r for r in rows if not r['gt_visible']]
            summary.append(dict(camera=camera,mode=mode,visible=len(pos),absent=len(neg),
                mean_visible_iou=float(np.mean([r['iou'] for r in pos])) if pos else None,
                valid_visible=sum(r['iou']>=.5 for r in pos),missed_visible=sum(not r['pred_visible'] for r in pos),
                absent_false_positive=sum(r['pred_visible'] for r in neg),abstained=sum(r['abstained'] for r in rows)))
    write_json(a.output/'scoring.json',dict(summary=summary,rows=results,meaning='Nonseed accepted masks, biased development sample'))
    write_json(a.output/'complete.json',dict(status='COMPLETE',cases=len(done),
        input_camera_frames=sum(c['frame_count'] for c in done),modes=list(MODES),
        case_seconds=sum(c['seconds'] for c in done),new_episode_ids=sorted({c['episode']['episode_id'] for c in cfg['cases'] if c['kind']=='unseen_episode'}),
        scored_labels=len(results)//4,config_sha256=sha(a.output/'run_config.json')))
    print(json.dumps(summary,indent=2))


def original_frame(case,idx):
    cap=cv2.VideoCapture(case['episode']['cameras'][case['camera']]['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
    ok,image=cap.read();cap.release();assert ok;return image


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run','score'])
    p.add_argument('--bank',type=Path,default=PROJECT/'annotations/identity_seed_bank_20260918_curated')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--eligible-first',action='store_true');p.add_argument('--scoring-only',action='store_true')
    p.add_argument('--candidate-parent',type=Path)
    a=p.parse_args();assert 0<=a.shard<a.shards
    {'prepare':prepare,'run':run,'score':score}[a.phase](a)
