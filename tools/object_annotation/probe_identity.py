"""Compare grounding identity prompts on observed production failures."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from annotate import BASE, write_json
from detector import Detector
from data import frames


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((a.root/'run_config.json').read_text())
    eps={e['episode_id']:e for e in json.loads((a.root/'episodes.json').read_text())['episodes']}
    detailed={o['object_id']:o for o in json.loads((a.root/'config/objects.json').read_text())}
    short={o['object_id']:o for o in json.loads((BASE/'config/objects.json').read_text())}
    negative={1000:dict(object_id=1000,class_name='robot arm',detailed_text_prompt='a robot arm'),
              1001:dict(object_id=1001,class_name='robot gripper',detailed_text_prompt='a black robot gripper'),
              1002:dict(object_id=1002,class_name='container',detailed_text_prompt='a plastic container'),
              1003:dict(object_id=1003,class_name='table',detailed_text_prompt='a green table')}
    torch.set_num_threads(4);cv2.setNumThreads(1)
    d=Detector(cfg['dino_config'],cfg['dino_checkpoint'],cfg['bert'],detailed)
    requested=[('episode_001684','head',0,[1]),('episode_001684','left_wrist',0,[1]),
               ('episode_002577','right_wrist',60,[2,4,10,20,21]),
               ('episode_002577','right_wrist',209,[2,4,10,20,21])]
    report=[]
    for ep,cam,frame,ids in requested:
        image=next(im for idx,_,im in frames(eps[ep],cam) if idx==frame)
        panels=[]
        for name,objects,query in [('detailed',detailed,ids),('short',short,ids),
                                   ('detailed_competitive',detailed|negative,ids+list(negative)),
                                   ('short_competitive',short|negative,ids+list(negative))]:
            d.objects=copy.deepcopy(objects);d.prompts.clear()
            detections,_=d.detect(image,query,floor=.30)
            report.append(dict(episode=ep,camera=cam,frame_idx=frame,mode=name,detections=detections))
            panel=image.copy()
            for r in detections:
                x0,y0,x1,y1=map(int,r['bbox_xyxy']);oid=r['object_id']
                color=tuple(map(int,np.random.default_rng(oid).integers(50,255,3)))
                cv2.rectangle(panel,(x0,y0),(x1,y1),color,2)
                cv2.putText(panel,f'{oid}:{r["confidence"]:.2f}',(x0,max(15,y0)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
            panel=cv2.resize(panel,(640,360))
            panel=np.pad(panel,((25,0),(0,0),(0,0)))
            cv2.putText(panel,name,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            panels.append(panel)
        sheet=np.concatenate([np.concatenate(panels[:2],1),np.concatenate(panels[2:],1)],0)
        cv2.imwrite(str(a.output/f'{ep}_{cam}_{frame:06d}.jpg'),sheet)
    write_json(a.output/'report.json',report)
    print('completed',flush=True)


if __name__=='__main__': main()
