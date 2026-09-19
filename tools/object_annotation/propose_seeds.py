"""Propose DINO seed candidates for visual review; never silently approve them."""
import argparse
import json
from pathlib import Path

import cv2

from annotate import draw,write_json
from data import frames


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('episode_id')
    p.add_argument('output',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    ep=next(e for e in json.loads((args.run/'selected_episodes.json').read_text())['episodes']
            if e['episode_id']==args.episode_id)
    source=args.run/f'task_{ep["task_id"]:02d}'/ep['episode_id'];seeds=[]
    for camera in ep['cameras']:
        candidates=[json.loads(x) for x in (source/f'{camera}.candidates.jsonl').read_text().splitlines()]
        chosen=[max([d for d in candidates if d['object_id']==oid],key=lambda d:d['confidence'])
                for oid in sorted({d['object_id'] for d in candidates})]
        lookup={idx:image for idx,_,image in frames(ep,camera) if idx in {d['frame_idx'] for d in chosen}}
        panels=[]
        for d in chosen:
            panel=draw(lookup[d['frame_idx']],[d],f'{camera} object={d["object_id"]} frame={d["frame_idx"]}')
            cv2.imwrite(str(args.output/f'{camera}_object_{d["object_id"]}.jpg'),panel);panels.append(panel)
            seeds.append(d|dict(review_status='unreviewed',source_detection_file=str((source/f'{camera}.candidates.jsonl').resolve())))
        cv2.imwrite(str(args.output/f'{camera}_sheet.jpg'),cv2.vconcat(panels))
    write_json(args.output/'proposed_seeds.json',seeds)


if __name__=='__main__':main()
