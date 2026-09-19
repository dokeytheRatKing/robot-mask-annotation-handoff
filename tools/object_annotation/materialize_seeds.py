"""Materialize explicitly reviewed frame/object choices from real DINO outputs."""
import argparse
import json
from pathlib import Path

from annotate import write_json

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('review',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();r=json.loads(a.review.read_text())
ep=next(e for e in json.loads((a.run/'selected_episodes.json').read_text())['episodes'] if e['episode_id']==r['episode_id'])
seeds=[]
for camera in ep['cameras']:
    file=a.run/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{camera}.candidates.jsonl'
    ds=[json.loads(x) for x in file.read_text().splitlines()]
    for oid,idx in r[camera].items():
        d=max([x for x in ds if x['object_id']==int(oid) and x['frame_idx']==idx],key=lambda x:x['confidence'])
        seeds.append(d|dict(review_status='accepted_for_pilot',reviewer=r['reviewer'],source_detection_file=str(file.resolve())))
write_json(a.output,seeds)
