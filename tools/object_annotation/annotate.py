#!/usr/bin/env python3
"""Task-conditioned three-camera bbox annotation; all outputs are independent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import cv2
import numpy as np

from data import CAMERAS,scan,select,frames
from diagnostics import compare

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def line(file,row):
    file.write(json.dumps(row,allow_nan=False)+'\n')


def draw(image,detections,title):
    overlay=image.copy();h,w=image.shape[:2]
    for d in detections:
        color=tuple(int(x) for x in np.random.default_rng(d['object_id']+10).integers(60,240,3))
        x0,y0,x1,y1=[int(round(x)) for x in d['bbox_xyxy']]
        cv2.rectangle(overlay,(x0,y0),(x1,y1),color,2)
        cv2.putText(overlay,f"{d['class_name']} {d['confidence']:.2f}",(x0,max(20,y0-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
    left=cv2.resize(image,(640,360));right=cv2.resize(overlay,(640,360))
    canvas=np.concatenate([left,right],axis=1)
    canvas=np.pad(canvas,((32,0),(0,0),(0,0)))
    cv2.putText(canvas,'RGB | '+title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return canvas


class QAVideo:
    def __init__(self,path,fps):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.temp=self.path.with_suffix('.partial.mp4')
        self.proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pixel_format','bgr24','-video_size','1280x392','-framerate',str(fps),
            '-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','22',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(self.temp)],stdin=subprocess.PIPE)
    def add(self,image):self.proc.stdin.write(image.tobytes())
    def close(self):
        self.proc.stdin.close()
        if self.proc.wait()!=0: raise RuntimeError('QA video encoder failed')
        self.temp.replace(self.path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='LeRobot root, image folder, or JSON manifest')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--objects',type=Path,default=BASE/'config/objects.json')
    p.add_argument('--task-objects',type=Path,default=BASE/'config/task_objects.json')
    p.add_argument('--prompt-overrides',type=Path,help='Versioned object_id -> detailed English description overrides')
    p.add_argument('--tasks',type=int,nargs='+',default=[1,9,24])
    p.add_argument('--episodes-per-task',type=int,default=5,help='0 selects all requested tasks episodes')
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--max-frames',type=int,default=0,help='Smoke-only limit per camera; 0 means complete stream')
    p.add_argument('--threshold',type=float,default=.30)
    p.add_argument('--candidate-floor',type=float,default=.15)
    p.add_argument('--nms-iou',type=float,default=.5)
    p.add_argument('--size',type=int,default=800)
    p.add_argument('--qa-per-task',type=int,default=2)
    p.add_argument('--checkpoint',default=str(PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'))
    p.add_argument('--model-config',default=str(PROJECT/'third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'))
    p.add_argument('--bert',default=str(PROJECT/'models/groundingdino/bert-base-uncased'))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    assert args.stride>0 and 0<args.candidate_floor<=args.threshold<1
    cv2.setNumThreads(1)
    objects={o['object_id']:o for o in json.loads(args.objects.read_text())}
    if args.prompt_overrides:
        for oid,description in json.loads(args.prompt_overrides.read_text()).items():
            objects[int(oid)]['detailed_text_prompt']=description
    mapping=json.loads(args.task_objects.read_text())
    for ids in mapping.values():assert set(ids)<=objects.keys()
    episodes=select(scan(args.input),args.tasks,args.episodes_per_task,args.seed)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
            if k not in ('resume','dry_run')}
    config.update(objects_sha256=sha(args.objects),task_objects_sha256=sha(args.task_objects),
                  schema='astribot.object_bbox.v1',camera_aliases=CAMERAS,
                  timestamp_semantics='original camera timestamp seconds, or null if unavailable',
                  confidence_semantics='mean sigmoid similarity of explicit noun-phrase tokens; not calibrated probability',
                  precision='float32',text_threshold=None,scoring='official positive-map token-span mean',
                  missing_policy='no detection row; processed frame ledger lists absent object IDs')
    if args.prompt_overrides:config['prompt_overrides_sha256']=sha(args.prompt_overrides)
    if args.dry_run:
        print(json.dumps(dict(config=config,episodes=episodes),indent=2));return
    # Never write annotation output inside any input dataset/source tree.
    dest=args.output.resolve();source=Path(args.input).resolve()
    if source.is_dir() and (dest==source or source in dest.parents):
        raise ValueError('Output must be separate from the input dataset')
    config['checkpoint_sha256']=sha(args.checkpoint)
    config['model_config_sha256']=sha(args.model_config)
    config['bert_files_sha256']={path.name:sha(path) for path in Path(args.bert).iterdir() if path.is_file()}
    config['groundingdino_revision']=subprocess.check_output(['git','-C',str(PROJECT/'third_party/GroundingDINO'),
                                                            'rev-parse','HEAD'],text=True).strip()
    config['script_hashes']={path.name:sha(path) for path in BASE.glob('*.py')}
    config['selected_manifest_sha256']=hashlib.sha256(json.dumps(episodes,sort_keys=True).encode()).hexdigest()
    inputroot=Path(args.input)
    if (inputroot/'meta/source_manifest.json').exists():
        config['source_manifest_sha256']=sha(inputroot/'meta/source_manifest.json')
    if args.resume:
        old=json.loads((dest/'run_config.json').read_text())
        if old!=config:raise ValueError('Resume config/provenance mismatch; use a new output directory')
    else:
        dest.mkdir(parents=True,exist_ok=False);write_json(dest/'run_config.json',config)
        write_json(dest/'selected_episodes.json',dict(episodes=episodes))
        (dest/'code').mkdir();(dest/'config').mkdir()
        for path in BASE.glob('*.py'):shutil.copy2(path,dest/'code'/path.name)
        shutil.copy2(args.objects,dest/'config/objects.json')
        shutil.copy2(args.task_objects,dest/'config/task_objects.json')
        write_json(dest/'config/resolved_objects.json',list(objects.values()))
        if args.prompt_overrides:shutil.copy2(args.prompt_overrides,dest/'config/prompt_overrides.json')
    from detector import Detector
    import torch
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    detector=Detector(args.model_config,args.checkpoint,args.bert,objects,size=args.size)
    write_json(dest/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        packages=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines()))
    rng=random.Random(args.seed+1);qa=set()
    for task in args.tasks:
        candidates=[e for e in episodes if e['task_id']==task]
        qa.update((e['task_id'],e['episode_id']) for e in rng.sample(candidates,min(args.qa_per_task,len(candidates))))
    torch.cuda.reset_peak_memory_stats();runstart=time.perf_counter();total=0
    for ep in episodes:
        ids=mapping[str(ep['task_id'])]
        for camera in CAMERAS:
            folder=dest/f'task_{ep["task_id"]:02d}'/ep['episode_id'];folder.mkdir(parents=True,exist_ok=True)
            marker=folder/f'{camera}.complete.json'
            if marker.exists():continue
            outputs={kind:folder/f'{camera}{suffix}.jsonl' for kind,suffix in
                     [('detections',''),('candidates','.candidates'),('frames','.frames'),('diagnostics','.diagnostics')]}
            handles={k:path.with_suffix('.jsonl.partial').open('w') for k,path in outputs.items()}
            writer=QAVideo(dest/'qa'/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{camera}.mp4',
                           ep.get('fps',30)/args.stride) if (ep['task_id'],ep['episode_id']) in qa else None
            start=time.perf_counter();model_seconds=0;count=0;ndet=0;previous=[];previous_idx=None
            try:
                for idx,ts,image in frames(ep,camera,args.stride,args.max_frames):
                    h,w=image.shape[:2];base=dict(episode_id=ep['episode_id'],task_id=ep['task_id'],
                        frame_idx=idx,timestamp=ts,camera=camera,image_width=w,image_height=h)
                    candidates,elapsed=detector.detect(image,ids,args.candidate_floor,args.nms_iou)
                    current=[d for d in candidates if d['confidence']>=args.threshold]
                    for d in candidates:line(handles['candidates'],base|d)
                    for d in current:line(handles['detections'],base|d)
                    line(handles['frames'],base|dict(object_ids=ids,detection_count=len(current),
                        absent_object_ids=[oid for oid in ids if not any(d['object_id']==oid for d in current)],
                        inference_seconds=elapsed))
                    if previous_idx is not None:
                        for d in compare(previous,current,idx,previous_idx,ids,w,h):
                            line(handles['diagnostics'],base|d)
                    if writer:
                        canvas=draw(image,current,f'{ep["episode_id"]} {camera} frame={idx} thr={args.threshold}')
                        writer.add(canvas)
                        if idx==0:cv2.imwrite(str(folder/f'{camera}.preview.jpg'),canvas)
                    previous=current;previous_idx=idx;model_seconds+=elapsed;count+=1;ndet+=len(current)
                for f in handles.values():f.close()
                if writer:writer.close()
                for k,path in outputs.items():path.with_suffix('.jsonl.partial').replace(path)
                seconds=time.perf_counter()-start;total+=count
                result=dict(episode_id=ep['episode_id'],task_id=ep['task_id'],camera=camera,
                    processed_frames=count,detections=ndet,wall_seconds=seconds,model_seconds=model_seconds,
                    images_per_second=count/seconds,model_images_per_second=count/model_seconds,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    files_sha256={path.name:sha(path) for path in outputs.values()})
                write_json(marker,result);print(json.dumps(result),flush=True)
            finally:
                for f in handles.values():f.close()
                if writer and writer.proc.poll() is None:
                    writer.proc.stdin.close();writer.proc.wait()
    write_json(dest/'run_complete.json',dict(episodes=len(episodes),cameras=3,
        newly_processed_images=total,wall_seconds=time.perf_counter()-runstart))


if __name__=='__main__':main()
