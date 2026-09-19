#!/usr/bin/env python3
"""Extract bounded RGB/object-only review images from completed full-run streams."""
import argparse
import gzip
import json
from pathlib import Path

import cv2

from annotate import write_json
import full_queue as queue
from full_segmentation import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--limit',default=16,type=int)
    p.add_argument('--task',type=int)
    p.add_argument('--episode')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    episodes={e['episode_id']:e for e in json.loads((a.root/'episodes.json').read_text())['episodes']}
    with queue.connect(a.root) as db:
        jobs=[json.loads(r['result']) for r in db.execute("SELECT result FROM jobs WHERE status='done' ORDER BY updated DESC")]
    result=[]
    for job in jobs:
        if a.task is not None and job['task_id']!=a.task:continue
        if a.episode and job['episode_id']!=a.episode:continue
        with gzip.open(a.root/job['files']['review'],'rt') as f: events=list(map(json.loads,f))
        ranked=sorted(events,key=lambda e:-sum(4 if 'no_identity_seed' in x['reasons'] else 1 for x in e['issues']))
        choices=[]
        for event in ranked:
            if all(abs(event['frame_idx']-x['frame_idx'])>=60 for x in choices):choices.append(event)
            if len(choices)>=2:break
        if not choices:choices=[dict(frame_idx=job['frames']//2,issues=[])]
        selected={e['frame_idx']:[] for e in choices}
        with gzip.open(a.root/job['files']['objects'],'rt') as f:
            for row in map(json.loads,f):
                if row['frame_idx'] in selected:selected[row['frame_idx']].append(row)
        ep=episodes[job['episode_id']]
        cap=cv2.VideoCapture(ep['cameras'][job['camera']]['video'])
        try:
            for event in choices:
                idx=event['frame_idx'];cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,image=cap.read()
                assert ok
                name=f'{job["episode_id"]}_{job["camera"]}_{idx:06d}'
                cv2.imwrite(str(a.output/f'{name}.jpg'),render(image,selected[idx],None,
                    f'task {job["task_id"]} {job["episode_id"]} {job["camera"]} frame {idx}: RGB / objects'))
                record=dict(task_id=job['task_id'],episode_id=job['episode_id'],camera=job['camera'],frame_idx=idx,
                    image=f'{name}.jpg',issues=event['issues'],source=job['files']['objects'],predictions=selected[idx])
                write_json(a.output/f'{name}.json',record);result.append(record)
                if len(result)>=a.limit:break
        finally:cap.release()
        if len(result)>=a.limit:break
    write_json(a.output/'index.json',dict(frames=result,policy='Assistant review first; not a request for human annotation'))
    print(json.dumps(dict(frames=len(result),output=str(a.output)),indent=2))


if __name__=='__main__':main()
