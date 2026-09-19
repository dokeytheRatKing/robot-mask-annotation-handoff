"""Matched-frame baseline / concrete-description comparison; derived images only."""
import argparse
import json
from pathlib import Path

import cv2
import torch

from annotate import BASE,PROJECT,write_json,draw
from data import frames
from detector import Detector


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    eps=json.loads(args.manifest.read_text())['episodes']
    selected=[next(e for e in eps if e['task_id']==task) for task in [1,9,24]]
    objects={x['object_id']:x for x in json.loads((BASE/'config/objects.json').read_text())}
    mapping=json.loads((BASE/'config/task_objects.json').read_text())
    overrides=json.loads((BASE/'config/prompt_overrides_visual_v2.json').read_text())
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model=Detector(str(PROJECT/'third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
                   str(PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'),
                   str(PROJECT/'models/groundingdino/bert-base-uncased'),objects)
    records=[]
    for ep in selected:
        wanted={0,(ep['frames']//2//5)*5,((ep['frames']-1)//5)*5}
        for camera in ep['cameras']:
            for idx,ts,image in frames(ep,camera):
                if idx not in wanted:continue
                pics=[]
                for version in ['document','visual_v2']:
                    for o in objects.values():
                        d=next(x for x in json.loads((BASE/'config/objects.json').read_text()) if x['object_id']==o['object_id'])
                        o['detailed_text_prompt']=overrides.get(str(o['object_id']),d['detailed_text_prompt']) if version=='visual_v2' else d['detailed_text_prompt']
                    model.prompts.clear();detections,elapsed=model.detect(image,mapping[str(ep['task_id'])])
                    records.append(dict(episode_id=ep['episode_id'],task_id=ep['task_id'],camera=camera,
                        frame_idx=idx,timestamp=ts,version=version,detections=detections,seconds=elapsed))
                    pics.append(draw(image,[d for d in detections if d['confidence']>=.30],f'{version} {camera} frame={idx}'))
                cv2.imwrite(str(args.output/f'{ep["episode_id"]}_{camera}_{idx:06d}.jpg'),cv2.vconcat(pics))
    write_json(args.output/'results.json',records)


if __name__=='__main__':main()
