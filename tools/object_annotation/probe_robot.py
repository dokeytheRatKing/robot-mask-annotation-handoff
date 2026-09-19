"""Task-independent robot-layer seed proposals, kept separate from object detections."""
import argparse
import json
from pathlib import Path

import cv2
import torch

from annotate import BASE,PROJECT,draw,write_json
from detector import Detector


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    objects={x['object_id']:x for x in json.loads((BASE/'config/robot_objects.json').read_text())}
    torch.set_num_threads(4);cv2.setNumThreads(1)
    detector=Detector(str(PROJECT/'third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
        str(PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'),
        str(PROJECT/'models/groundingdino/bert-base-uncased'),objects)
    result=[]
    for metadata in sorted((a.cache_run/'frame_cache').glob('*/*/frame_map.json')):
        frames=json.loads(metadata.read_text())
        for local in [0,30,60,90,125,160,200,len(frames)-1]:
            if local>=len(frames):continue
            image=cv2.imread(str(metadata.parent/f'{local:06d}.jpg'))
            boxes,sec=detector.detect(image,list(objects),floor=.15)
            base=frames[local]
            result.extend(base|d for d in boxes)
            canvas=draw(image,[d for d in boxes if d['confidence']>=.25],
                f'robot proposal {base["camera"]} frame={base["frame_idx"]} thr=0.25')
            cv2.imwrite(str(a.output/f'{base["camera"]}_{base["frame_idx"]:06d}.jpg'),canvas)
    write_json(a.output/'robot_candidates.json',result)


if __name__=='__main__':main()
