#!/usr/bin/env python3
"""Object + separate robot segmentation pilot from the existing reviewed SAM2 run."""
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

from annotate import BASE,PROJECT,sha,write_json,line,QAVideo
from masks import encode,decode,record,overlay


def propagate(predictor,state,seed,metadata):
    index_by_frame={m['frame_idx']:i for i,m in enumerate(metadata)}
    start=index_by_frame[seed['frame_idx']];result={}
    for reverse in [False,True]:
        if reverse and start==0:continue
        predictor.reset_state(state)
        kwargs={}
        if 'points' in seed:
            kwargs.update(points=np.asarray(seed['points'],np.float32),labels=np.asarray(seed['labels'],np.int32))
        predictor.add_new_points_or_box(state,frame_idx=start,obj_id=1,
                                       box=np.asarray(seed['box'],np.float32),**kwargs)
        for local,_,logits in predictor.propagate_in_video(state,start_frame_idx=start,reverse=reverse):
            if reverse and local==start:continue
            binary=(logits[0,0]>0).cpu().numpy()
            outputs=state['output_dict_per_obj'][0]
            current=outputs['cond_frame_outputs'].get(local)
            if current is None:current=outputs['non_cond_frame_outputs'][local]
            presence=float(current['object_score_logits'].float().sigmoid().reshape(-1)[0])
            conf=presence if seed.get('confidence') is None else seed['confidence']*presence
            result[local]=record(binary,confidence=conf,sam2_presence=presence,
                confidence_source='sam2_presence' if seed.get('confidence') is None else 'dino_seed_score_times_sam2_presence',
                seed_frame_idx=seed['frame_idx'])
    assert set(result)==set(range(len(metadata))), 'Propagation missed frame(s)'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--robot-prompts',type=Path,default=BASE/'config/robot_mask_prompts_pilot.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cameras',nargs='+',default=['head','left_wrist','right_wrist'])
    a=p.parse_args()
    source=a.source_run.resolve();dest=a.output.resolve()
    if source==dest or source in dest.parents:raise ValueError('Use independent output directory')
    srcconfig=json.loads((source/'run_config.json').read_text())
    episodes=json.loads((source/'selected_episodes.json').read_text())['episodes']
    assert len(episodes)==1,'Bounded pilot only'
    ep=episodes[0];robot_prompts=json.loads(a.robot_prompts.read_text())
    assert ep['episode_id']==robot_prompts['episode_id']
    seeds=json.loads((source/'reviewed_seeds.json').read_text())
    assert all(s['review_status']=='accepted_for_pilot' for s in seeds)
    mapping=json.loads((BASE/'config/task_objects.json').read_text())
    objects={x['object_id']:x for x in json.loads((BASE/'config/objects.json').read_text())}
    dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'selected_episodes.json',dict(episodes=episodes))
    write_json(dest/'run_config.json',dict(schema='astribot.segmentation.v1',source_run=str(source),
        source_run_config_sha256=sha(source/'run_config.json'),source_seeds_sha256=sha(source/'reviewed_seeds.json'),
        robot_prompts_sha256=sha(a.robot_prompts),checkpoint=srcconfig['sam2_checkpoint'],
        checkpoint_sha256=sha(srcconfig['sam2_checkpoint']),sam2_revision=srcconfig['sam2_revision'],
        stride=srcconfig['stride'],cameras=a.cameras,precision='bfloat16 autocast',mask_threshold_logit=0,
        visibility_semantics='visible=true iff predicted mask nonempty; false=predicted empty; null=uninitialized, not ground-truth visibility',
        confidence_semantics='objects: DINO seed score * SAM2 presence; robot component: SAM2 presence; robot union: minimum presence of nonempty contributors (0 if all empty). Heuristics, not calibrated mask quality.',
        identity_semantics='robot components are camera-local; no asserted anatomical left/right identity',
        overlap_policy='independent masks; never subtract robot from object, never enforce mutual exclusion',
        script_sha256=sha(__file__),mask_helpers_sha256=sha(BASE/'masks.py')))
    shutil.copy2(a.robot_prompts,dest/'robot_prompts.json');shutil.copy2(source/'reviewed_seeds.json',dest/'object_seeds.json')
    shutil.copy2(__file__,dest/'mask_pilot_snapshot.py');shutil.copy2(BASE/'masks.py',dest/'masks_snapshot.py')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml',srcconfig['sam2_checkpoint'],device='cuda')
    allmetrics=[]
    for camera in a.cameras:
        cache=source/'frame_cache'/ep['episode_id']/camera
        metadata=json.loads((cache/'frame_map.json').read_text())
        assert len(metadata)==len(range(0,ep['frames'],srcconfig['stride']))
        selected=[s for s in seeds if s['camera']==camera]
        components=[s for s in robot_prompts['components'] if s['camera']==camera]
        assert components and len({s['component_id'] for s in components})==len(components)
        folder=dest/f'task_{ep["task_id"]:02d}'/ep['episode_id'];folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/f'{camera}.frame_cache_hashes.json',{path.name:sha(path) for path in sorted(cache.glob('*')) if path.is_file()})
        outputs={};robots={};start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=predictor.init_state(str(cache),offload_video_to_cpu=True,offload_state_to_cpu=True)
            for s in selected:
                outputs[s['object_id']]=propagate(predictor,state,s|dict(box=s['bbox_xyxy']),metadata)
                print(camera,'object',s['object_id'],'masks complete',flush=True)
            for s in components:
                robots[s['component_id']]=propagate(predictor,state,s,metadata)
                print(camera,'robot',s['component_id'],'masks complete',flush=True)
            del state
        torch.cuda.synchronize();model_seconds=time.perf_counter()-start
        files={k:folder/f'{camera}.{k}.jsonl' for k in ['objects','robot','robot_components']}
        handles={k:path.with_suffix('.jsonl.partial').open('w') for k,path in files.items()}
        qa=QAVideo(dest/'qa'/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{camera}.mp4',ep['fps']/srcconfig['stride'])
        robotqa=QAVideo(dest/'qa_robot'/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{camera}.mp4',ep['fps']/srcconfig['stride'])
        visible_objects=0;expected_ids=mapping[str(ep['task_id'])]
        try:
            for local,base in enumerate(metadata):
                rs=[];os=[]
                for oid in expected_ids:
                    if oid in outputs:
                        result=outputs[oid][local]
                    else:result=dict(mask=None,bbox_xyxy=None,mask_area=None,visible=None,confidence=None,
                                     mask_status='not_initialized',confidence_source=None)
                    row=base|dict(object_id=oid,class_name=objects[oid]['class_name'],layer='object')|result
                    line(handles['objects'],row);os.append(row);visible_objects+=row['visible'] is True
                union=np.zeros((base['image_height'],base['image_width']),bool)
                for s in components:
                    row=base|dict(object_id=s['component_id'],class_name=s['class_name'],layer='robot_component')|robots[s['component_id']][local]
                    line(handles['robot_components'],row);rs.append(row);union|=decode(row['mask'])
                scores=[r['confidence'] for r in rs if r['visible']]
                robot=base|dict(object_id='robot_mask',class_name='robot',layer='robot')|record(union,
                    confidence=min(scores) if scores else 0.,confidence_source='min_presence_of_nonempty_components',
                    component_ids=[s['component_id'] for s in components])
                line(handles['robot'],robot)
                image=cv2.imread(str(cache/f'{local:06d}.jpg'))
                canvas=overlay(image,os,robot,f'{camera} frame={base["frame_idx"]}')
                qa.add(canvas);robotcanvas=overlay(image,os,robot,f'{camera} frame={base["frame_idx"]}',mode='robot')
                robotqa.add(robotcanvas)
                if local%10==0 or local==len(metadata)-1:
                    out=dest/'qa_frames'/camera;out.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(out/f'{base["frame_idx"]:06d}.jpg'),canvas)
                    cv2.imwrite(str(out/f'{base["frame_idx"]:06d}.robot.jpg'),robotcanvas)
            for h in handles.values():h.close()
            qa.close();robotqa.close()
            for path in files.values():path.with_suffix('.jsonl.partial').replace(path)
        finally:
            for h in handles.values():h.close()
            for video in [qa,robotqa]:
                if video.proc.poll() is None:video.proc.stdin.close();video.proc.wait()
        metric=dict(camera=camera,sampled_images=len(metadata),objects=len(expected_ids),seeded_objects=len(selected),
            robot_components=len(components),visible_object_rows=visible_objects,
            model_seconds=model_seconds,wall_seconds=time.perf_counter()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            files_sha256={path.name:sha(path) for path in files.values()})
        write_json(folder/f'{camera}.complete.json',metric);allmetrics.append(metric);print(json.dumps(metric),flush=True)
    write_json(dest/'run_complete.json',dict(status='COMPLETE_PENDING_QUALITY_REVIEW',streams=allmetrics))


if __name__=='__main__':main()
