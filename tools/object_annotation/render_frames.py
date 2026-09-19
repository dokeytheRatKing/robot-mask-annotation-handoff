"""Render specific QA frames from a run without rerunning detection."""
import argparse
import json
from pathlib import Path

import cv2

from annotate import draw
from data import frames

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('episode_id')
p.add_argument('camera');p.add_argument('indices',type=int,nargs='+');p.add_argument('--floor',type=float,default=.15)
a=p.parse_args();ep=next(e for e in json.loads((a.run/'selected_episodes.json').read_text())['episodes'] if e['episode_id']==a.episode_id)
folder=a.run/f'task_{ep["task_id"]:02d}'/ep['episode_id']
path=folder/f'{a.camera}.candidates.jsonl'
if not path.exists():path=folder/f'{a.camera}.jsonl'
rows=[json.loads(x) for x in path.read_text().splitlines()]
out=a.run/'qa_frames';out.mkdir(exist_ok=True)
for idx,ts,image in frames(ep,a.camera):
    if idx not in a.indices:continue
    ds=[r for r in rows if r['frame_idx']==idx and r['confidence']>=a.floor]
    cv2.imwrite(str(out/f'{a.episode_id}_{a.camera}_{idx:06d}.jpg'),draw(image,ds,f'{a.episode_id} {a.camera} frame={idx} floor={a.floor}'))
