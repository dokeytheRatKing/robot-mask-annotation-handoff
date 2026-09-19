#!/usr/bin/env python3
"""GroundingDINO-keyframe / official SAM2 propagation. Identical required bbox fields.

Uses reviewed *detector* boxes, not fabricated seeds or highest-score assumptions.
Does not emit masks. Empty predicted masks produce no detection rows.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

from annotate import PROJECT,BASE,write_json,line,sha,draw,QAVideo
from data import frames
from diagnostics import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dino-run',type=Path,required=True)
    p.add_argument('--seeds',type=Path,required=True,help='Reviewed list with source_detection_file and accepted_for_pilot status')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--checkpoint',type=Path,default=PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt')
    p.add_argument('--model-config',default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    args=p.parse_args();assert args.stride>0
    seeds=json.loads(args.seeds.read_text())
    assert seeds and all(s['review_status']=='accepted_for_pilot' for s in seeds)
    manifest=json.loads((args.dino_run/'selected_episodes.json').read_text())['episodes']
    selected=[e for e in manifest if e['episode_id'] in {s['episode_id'] for s in seeds}]
    mapping=json.loads((BASE/'config/task_objects.json').read_text())
    ids=set()
    for s in seeds:
        assert s['object_id'] in mapping[str(s['task_id'])]
        key=(s['episode_id'],s['camera'],s['object_id']);assert key not in ids;ids.add(key)
        source=[json.loads(x) for x in Path(s['source_detection_file']).read_text().splitlines()]
        assert any(all(r[k]==s[k] for k in ['episode_id','task_id','camera','frame_idx','object_id','bbox_xyxy','confidence']) for r in source), 'Seed is not a real DINO output'
        assert s['frame_idx']%args.stride==0,'Seed must fall on sampled frame grid'
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(schema='astribot.object_bbox.v1',backend='official_groundingdino_plus_official_sam2',
        stride=args.stride,sam2_checkpoint=str(args.checkpoint),sam2_checkpoint_sha256=sha(args.checkpoint),
        sam2_revision=subprocess.check_output(['git','-C',str(PROJECT/'third_party/sam2'),'rev-parse','HEAD'],text=True).strip(),
        seed_manifest_sha256=sha(args.seeds),dino_config_sha256=sha(args.dino_run/'run_config.json'),
        confidence_semantics='seed GroundingDINO confidence multiplied by SAM2 sigmoid object-presence score; heuristic, not calibrated or equivalent to single-frame DINO confidence',
        output_box_semantics='tight bounding box of positive SAM2 mask pixels; no box for empty mask',
        precision='bfloat16 autocast',directions='independent forward and reverse propagation per object',
        script_sha256=sha(__file__))
    write_json(args.output/'run_config.json',config);write_json(args.output/'selected_episodes.json',dict(episodes=selected))
    shutil.copy2(args.seeds,args.output/'reviewed_seeds.json')
    shutil.copy2(__file__,args.output/'sam2_refine_snapshot.py')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    predictor=build_sam2_video_predictor(args.model_config,str(args.checkpoint),device='cuda')
    metrics=[]
    for ep in selected:
        for camera in ep['cameras']:
            chosen=[s for s in seeds if s['episode_id']==ep['episode_id'] and s['camera']==camera]
            folder=args.output/f'task_{ep["task_id"]:02d}'/ep['episode_id'];folder.mkdir(parents=True,exist_ok=True)
            # Derived JPEG working cache, retained for reproducibility. Originals untouched.
            cache=args.output/'frame_cache'/ep['episode_id']/camera;cache.mkdir(parents=True)
            metadata=[]
            for local,(idx,ts,image) in enumerate(frames(ep,camera,args.stride)):
                cv2.imwrite(str(cache/f'{local:06d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,98])
                metadata.append(dict(episode_id=ep['episode_id'],task_id=ep['task_id'],frame_idx=idx,
                    timestamp=ts,camera=camera,image_width=image.shape[1],image_height=image.shape[0]))
            write_json(cache/'frame_map.json',metadata)
            result=defaultdict(list);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=predictor.init_state(str(cache),offload_video_to_cpu=True,offload_state_to_cpu=True)
                for seed in chosen:
                    local_seed=seed['frame_idx']//args.stride
                    for reverse in [False,True]:
                        if reverse and local_seed==0:continue
                        predictor.reset_state(state)
                        predictor.add_new_points_or_box(state,frame_idx=local_seed,obj_id=seed['object_id'],
                                                       box=np.asarray(seed['bbox_xyxy'],dtype=np.float32))
                        for local,oids,logits in predictor.propagate_in_video(state,start_frame_idx=local_seed,reverse=reverse):
                            if reverse and local==local_seed:continue
                            mask=logits[0,0]>0
                            coords=torch.where(mask)
                            if coords[0].numel()==0:continue
                            y,x=coords
                            outputs=state['output_dict_per_obj'][0]
                            current=outputs['cond_frame_outputs'].get(local)
                            if current is None:current=outputs['non_cond_frame_outputs'][local]
                            presence=float(current['object_score_logits'].float().sigmoid().reshape(-1)[0])
                            box=[float(x.min()),float(y.min()),float(x.max()+1),float(y.max()+1)]
                            result[local].append(dict(object_id=seed['object_id'],class_name=seed['class_name'],
                                bbox_xyxy=box,confidence=seed['confidence']*presence,
                                confidence_source='dino_seed_score_times_sam2_presence',
                                seed_confidence=seed['confidence'],sam2_presence=presence,
                                seed_frame_idx=seed['frame_idx'],backend='grounded_sam2'))
                    print(f'{ep["episode_id"]} {camera} object {seed["object_id"]} propagated',flush=True)
                del state
            torch.cuda.synchronize();inference_seconds=time.perf_counter()-start
            paths={k:folder/f'{camera}{s}.jsonl' for k,s in [('detections',''),('frames','.frames'),('diagnostics','.diagnostics')]}
            handles={k:path.open('w') for k,path in paths.items()}
            qa=QAVideo(args.output/'qa'/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{camera}.mp4',ep['fps']/args.stride)
            previous=[];previous_idx=None;required=mapping[str(ep['task_id'])]
            for local,base in enumerate(metadata):
                ds=result[local]
                for d in ds:line(handles['detections'],base|d)
                line(handles['frames'],base|dict(detection_count=len(ds),object_ids=required,
                    absent_object_ids=[oid for oid in required if not any(d['object_id']==oid for d in ds)],
                    seeded_object_ids=[s['object_id'] for s in chosen]))
                if previous_idx is not None:
                    for d in compare(previous,ds,base['frame_idx'],previous_idx,required,base['image_width'],base['image_height']):
                        line(handles['diagnostics'],base|d)
                image=cv2.imread(str(cache/f'{local:06d}.jpg'))
                canvas=draw(image,ds,f'SAM2 {ep["episode_id"]} {camera} frame={base["frame_idx"]}')
                qa.add(canvas)
                if local in [0,len(metadata)//2,len(metadata)-1]:cv2.imwrite(str(folder/f'{camera}.{local:06d}.preview.jpg'),canvas)
                previous=ds;previous_idx=base['frame_idx']
            for f in handles.values():f.close()
            qa.close()
            metric=dict(episode_id=ep['episode_id'],camera=camera,frames=len(metadata),seeded_objects=len(chosen),
                detections=sum(map(len,result.values())),inference_seconds=inference_seconds,
                images_per_second=len(metadata)/inference_seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                output_hashes={path.name:sha(path) for path in paths.values()})
            metrics.append(metric);write_json(folder/f'{camera}.complete.json',metric)
            print(json.dumps(metric),flush=True)
    write_json(args.output/'run_complete.json',dict(status='COMPLETE',streams=metrics))


if __name__=='__main__':main()
