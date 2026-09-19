"""Bounded causal A/B: identical initial seeds, uninterrupted SAM2 vs recovery.

Each detector call is fresh official GroundingDINO inference, shared across A/B.
Existing reviewed bidirectional output is an additional historical reference.
"""
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import sam2.sam2_video_predictor as video_module

from annotate import BASE,PROJECT,write_json,line,sha,QAVideo
from data import frames
from detector import Detector
from masks import record,decode,bbox
from recovery import Settings,choose_candidates,diagnostics,box_iou,terminate_reason


def fresh_state(template):
    shared={'images','cached_features','constants'}
    return {k:v if k in shared else copy.deepcopy(v) for k,v in template.items()}


def predict(predictor,state,local,seed=None):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        if seed is not None:
            predictor.reset_state(state)
            predictor.add_new_points_or_box(state,frame_idx=local,obj_id=1,box=np.asarray(seed,np.float32))
        result=list(predictor.propagate_in_video(state,start_frame_idx=local,max_frame_num_to_track=0))
        assert len(result)==1 and result[0][0]==local
        binary=(result[0][2][0,0]>0).cpu().numpy()
        out=state['output_dict_per_obj'][0]
        current=out['cond_frame_outputs'].get(local,out['non_cond_frame_outputs'].get(local))
        presence=float(current['object_score_logits'].float().sigmoid().reshape(-1)[0])
    return binary,presence


def panel(image,rows,robot,label):
    out=image.copy();union=np.zeros(image.shape[:2],bool)
    if robot is not None:out[robot]=(out[robot]*.65+np.array([255,210,0])*.35).astype(np.uint8)
    for row in rows:
        if row['mask'] is None or not row['visible']:continue
        m=decode(row['mask']);union|=m
        color=np.random.default_rng(row['object_id']+7).integers(65,245,3)
        out[m]=(out[m]*.5+color*.5).astype(np.uint8)
        x0,y0,x1,y1=row['bbox_xyxy'];cv2.rectangle(out,(x0,y0),(x1,y1),tuple(map(int,color)),1)
    if robot is not None:out[union&robot]=[255,0,255]
    out=cv2.resize(out,(640,360));out=np.pad(out,((32,0),(0,0),(0,0)))
    cv2.putText(out,label,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    text=' | '.join(f'{r["class_name"]}:{r["tracking_status"][:4]}' for r in rows)
    cv2.putText(out,text,(5,29),cv2.FONT_HERSHEY_SIMPLEX,.32,(240,240,240),1)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',nargs='+',default=['episode_002478','episode_001608','episode_002577'])
    p.add_argument('--cameras',nargs='+',default=['head','left_wrist','right_wrist'])
    p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=a.output
    cfg=Settings();stride=5
    dino_root=PROJECT/'annotations/groundingdino_pilot_20260916'
    old_root=PROJECT/'annotations/grounded_sam2_mask_pilot_20260916'
    episodes=[e for e in json.loads((dino_root/'selected_episodes.json').read_text())['episodes'] if e['episode_id'] in a.episodes]
    assert {e['episode_id'] for e in episodes}==set(a.episodes)
    ids_by_task=json.loads((BASE/'config/task_objects.json').read_text())
    objects={o['object_id']:o for o in json.loads((BASE/'config/objects.json').read_text())}
    dc=json.loads((dino_root/'run_config.json').read_text())
    sam=PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt'
    write_json(root/'selected_episodes.json',dict(episodes=episodes))
    write_json(root/'run_config.json',dict(schema='astribot.segmentation.recovery.v1',stride=stride,settings=asdict(cfg),
        cameras=a.cameras,max_frames=a.max_frames,initialization='same two-confirmation automatic initial seeds in both A/B arms; no future-frame access',
        source_manifest_sha256=dc['source_manifest_sha256'],sam2_checkpoint_sha256=sha(sam),
        dino_checkpoint_sha256=sha(dc['checkpoint']),object_config_sha256=sha(BASE/'config/objects.json'),
        robot_reference=str(old_root),robot_policy='QA and suspicious trigger only; never subtract; unavailable extra episodes stay null',
        confidence='last DINO seed score * current SAM2 presence, not calibrated mask quality',
        script_hashes={n:sha(BASE/n) for n in ['recovery.py','recovery_pilot.py','masks.py','detector.py','data.py']},
        precision='SAM2 BF16; GroundingDINO FP32',temporal_semantics='causal sampled frames; physical-instance identity unverified'))
    for name in ['recovery.py','recovery_pilot.py']:shutil.copy2(BASE/name,root/name)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    video_module.tqdm=lambda iterable,**kwargs:iterable
    detector=Detector(dc['model_config'],dc['checkpoint'],dc['bert'],objects,size=dc['size'])
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml',str(sam),device='cuda')
    all_stats=[]
    for ep in episodes:
        for camera in a.cameras:
            folder=root/f'task_{ep["task_id"]:02d}'/ep['episode_id'];folder.mkdir(parents=True,exist_ok=True)
            cache=root/'frame_cache'/ep['episode_id']/camera;cache.mkdir(parents=True)
            meta=[]
            for local,(idx,ts,image) in enumerate(frames(ep,camera,stride,a.max_frames)):
                assert cv2.imwrite(str(cache/f'{local:06d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,98])
                meta.append(dict(episode_id=ep['episode_id'],task_id=ep['task_id'],camera=camera,frame_idx=idx,
                    timestamp=ts,image_width=image.shape[1],image_height=image.shape[0]))
            write_json(cache/'frame_map.json',meta)
            robots={}
            oldfile=old_root/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{camera}.robot.jsonl'
            if oldfile.exists():robots={r['frame_idx']:r for r in map(json.loads,oldfile.read_text().splitlines())}
            streams={name:(folder/f'{camera}.{name}.jsonl').open('w') for name in ['baseline','recovery','robot','events','detections']}
            video=QAVideo(root/'qa'/ep['episode_id']/f'{camera}.mp4',ep['fps']/stride)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                template=predictor.init_state(str(cache),offload_video_to_cpu=True,offload_state_to_cpu=True)
            ids=ids_by_task[str(ep['task_id'])]
            tracks={oid:dict(baseline=None,recovery=None,initial=None,seed=None,segment=0,misses=0,empty=0,pending=None,last_detection=-999) for oid in ids}
            previous={};stats=Counter();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for local,base in enumerate(meta):
                image=cv2.imread(str(cache/f'{local:06d}.jpg'));h,w=image.shape[:2];zero=np.zeros((h,w),bool)
                rr=robots.get(base['frame_idx'],base|dict(object_id='robot_mask',class_name='robot',mask=None,
                    bbox_xyxy=None,confidence=None,visible=None,mask_status='not_initialized',layer='robot',provenance='not_available_for_extra_episode'))
                robot=decode(rr['mask']) if rr['mask'] is not None else None;line(streams['robot'],rr)
                pred={mode:{} for mode in ['baseline','recovery']}
                for oid,t in tracks.items():
                    for mode in pred:
                        pred[mode][oid]=predict(predictor,t[mode],local) if t[mode] is not None else (zero,0.)
                    t['empty']=0 if pred['recovery'][oid][0].any() else t['empty']+1
                flags=diagnostics({oid:x[0] for oid,x in pred['recovery'].items()},previous,robot,cfg)
                due=local%cfg.detection_period==0 or any(
                    local-t['last_detection']>=cfg.search_period and (flags[oid] or t['recovery'] is None) for oid,t in tracks.items())
                if due:
                    detections,seconds=detector.detect(image,ids,floor=.15)
                    stats['detector_calls']+=1;stats['detector_seconds']+=seconds
                    line(streams['detections'],base|dict(detections=detections,seconds=seconds,image_sha256=sha(cache/f'{local:06d}.jpg')))
                    candidates,rejections=choose_candidates(detections,ids,cfg)
                    for oid,t in tracks.items():
                        t['last_detection']=local;c=candidates.get(oid)
                        if oid in rejections:flags[oid].append(rejections[oid])
                        confirmed=False
                        if c:
                            old=t['pending']
                            confirmed=old is not None and 0<local-old['local']<=cfg.detection_period and box_iou(c['bbox_xyxy'],old['bbox_xyxy'])>=cfg.confirmation_iou
                            t['pending']=dict(local=local,bbox_xyxy=c['bbox_xyxy'])
                            t['misses']=0
                        else:t['pending']=None;t['misses']+=1
                        event=None
                        if c and confirmed:
                            if t['initial'] is None:
                                t['initial']=c|dict(frame_idx=base['frame_idx']);t['seed']=t['initial'];t['segment']=1
                                for mode in pred:
                                    t[mode]=fresh_state(template);pred[mode][oid]=predict(predictor,t[mode],local,c['bbox_xyxy'])
                                event='initial_seed_shared'
                            elif t['recovery'] is None or box_iou(bbox(pred['recovery'][oid][0]),c['bbox_xyxy'])<.30 or any(f not in ['robot_overlap_warning'] for f in flags[oid]):
                                event='reentry_reseed' if t['recovery'] is None else 'suspicious_reseed'
                                if t['recovery'] is None:t['recovery']=fresh_state(template)
                                t['seed']=c|dict(frame_idx=base['frame_idx']);t['segment']+=1
                                pred['recovery'][oid]=predict(predictor,t['recovery'],local,c['bbox_xyxy'])
                                t['empty']=0 if pred['recovery'][oid][0].any() else t['empty']
                        reason=terminate_reason(flags[oid],t['empty'],t['misses'],pred['recovery'][oid][1],cfg)
                        if reason and t['recovery'] is not None:
                            event='terminate';flags[oid].append(reason);t['recovery']=None;pred['recovery'][oid]=(zero,0.)
                        if event:
                            stats[event]+=1;line(streams['events'],base|dict(object_id=oid,event=event,segment=t['segment'],
                                flags=flags[oid],candidate=c,termination_reason=reason,raw_policy='no_robot_subtraction'))
                rendered={}
                for mode in pred:
                    rows=[]
                    for oid,t in tracks.items():
                        m,presence=pred[mode][oid];seed=t['initial'] if mode=='baseline' else t['seed']
                        row=base|dict(object_id=oid,class_name=objects[oid]['class_name'],layer='object',
                            track_id=f'{ep["episode_id"]}:{camera}:object_{oid}',segment_id=1 if mode=='baseline' else t['segment'],
                            seed_frame_idx=seed['frame_idx'] if seed else None,sam2_presence=presence,
                            suspicious_flags=flags[oid] if mode=='recovery' else [],
                            provenance=dict(mode=mode,seed=seed,automatic=True),
                            tracking_status='active' if t[mode] is not None else 'lost' if seed else 'uninitialized')
                        if seed:row.update(record(m,confidence=seed['confidence']*presence))
                        else:row.update(mask=None,bbox_xyxy=None,mask_area=None,confidence=None,visible=None,mask_status='not_initialized')
                        if mode=='recovery' and seed and t[mode] is None:row['mask_status']='terminated_pending_redetection'
                        line(streams[mode],row);rows.append(row)
                        stats[f'{mode}_nonempty']+=int(row['visible'] is True)
                    rendered[mode]=rows
                video.add(np.concatenate([panel(image,rendered[mode],robot,f'{mode} {camera} frame={base["frame_idx"]}') for mode in pred],axis=1))
                previous={oid:x[0] for oid,x in pred['recovery'].items()}
                if local%50==0:print(ep['episode_id'],camera,local,len(meta),dict(stats),flush=True)
            video.close()
            for f in streams.values():f.close()
            metric=dict(episode_id=ep['episode_id'],camera=camera,frames=len(meta),wall_seconds=time.perf_counter()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,counts=dict(stats),
                files_sha256={p.name:sha(p) for p in folder.glob(f'{camera}.*.jsonl')})
            write_json(folder/f'{camera}.complete.json',metric);all_stats.append(metric)
            write_json(folder/f'{camera}.initial_seeds.json',{str(oid):t['initial'] for oid,t in tracks.items()})
            del tracks,template;print(json.dumps(metric),flush=True)
    write_json(root/'run_complete.json',dict(status='COMPLETE',streams=all_stats,quality='UNMEASURED_WITHOUT_HUMAN_GT'))
    html=['<!doctype html><meta charset="utf-8"><title>Tracking recovery A/B</title><h1>Uninterrupted SAM2 / automatic recovery</h1>',
          '<p>Same automatic initial seeds. Left: control. Right: recovery. Cyan robot, magenta overlap. Model diagnostics are not GT accuracy.</p>']
    for s in all_stats:
        html.append(f'<h2>{s["episode_id"]} / {s["camera"]}</h2><video controls preload="none" width="1280" src="qa/{s["episode_id"]}/{s["camera"]}.mp4"></video>')
    (root/'qa_index.html').write_text('\n'.join(html))


if __name__=='__main__':main()
