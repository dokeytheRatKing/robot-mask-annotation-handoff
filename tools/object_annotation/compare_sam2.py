"""Verify fallback schema and compare consistency on identical sampled frames."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from annotate import sha,write_json
from diagnostics import geometry


def read(path):return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('dino',type=Path);p.add_argument('sam',type=Path);args=p.parse_args()
    eps=json.loads((args.sam/'selected_episodes.json').read_text())['episodes']
    seeds=json.loads((args.sam/'reviewed_seeds.json').read_text())
    metrics=[]
    for ep in eps:
        rel=Path(f'task_{ep["task_id"]:02d}')/ep['episode_id']
        for cam in ep['cameras']:
            marker=json.loads((args.sam/rel/f'{cam}.complete.json').read_text())
            for name,digest in marker['output_hashes'].items():assert sha(args.sam/rel/name)==digest
            dl=read(args.dino/rel/f'{cam}.frames.jsonl');sl=read(args.sam/rel/f'{cam}.frames.jsonl')
            assert [(x['frame_idx'],x['timestamp'],x['image_width'],x['image_height']) for x in dl]==[(x['frame_idx'],x['timestamp'],x['image_width'],x['image_height']) for x in sl]
            seeded={s['object_id'] for s in seeds if s['episode_id']==ep['episode_id'] and s['camera']==cam}
            for backend,root in [('dino',args.dino),('sam2',args.sam)]:
                ds=read(root/rel/f'{cam}.jsonl');diagnostics=read(root/rel/f'{cam}.diagnostics.jsonl')
                ledger={x['frame_idx']:x for x in sl}
                for d in ds:
                    x0,y0,x1,y1=d['bbox_xyxy'];assert 0<=x0<x1<=d['image_width'] and 0<=y0<y1<=d['image_height']
                    assert np.isfinite(d['confidence']) and 0<=d['confidence']<=1
                    assert d['timestamp']==ledger[d['frame_idx']]['timestamp'] and d['task_id']==ep['task_id'] and d['episode_id']==ep['episode_id'] and d['camera']==cam
                    if backend=='sam2':assert d['object_id'] in seeded
                for oid in dl[0]['object_ids']:
                    d=[x for x in diagnostics if x['object_id']==oid]
                    scores=[x['iou'] for x in d if 'iou' in x]
                    metrics.append(dict(episode_id=ep['episode_id'],camera=cam,object_id=oid,backend=backend,
                        seeded=oid in seeded,frames=len(sl),detected_frames=len({x['frame_idx'] for x in ds if x['object_id']==oid}),
                        suspicious=sum(x['suspicious'] for x in d),comparisons=len(d),
                        median_iou=float(np.median(scores)) if scores else None,
                        low_iou=sum('low_iou' in x['reasons'] for x in d),
                        center_jump=sum('center_jump' in x['reasons'] for x in d),
                        area_jump=sum('area_jump' in x['reasons'] for x in d)))
    with (args.sam/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    summary={}
    for cam in ['head','left_wrist','right_wrist']:
        summary[cam]={}
        for backend in ['dino','sam2']:
            m=[x for x in metrics if x['camera']==cam and x['backend']==backend and x['seeded']]
            summary[cam][backend]=dict(seeded_object_frame_opportunities=sum(x['frames'] for x in m),
                detected_frames=sum(x['detected_frames'] for x in m),suspicious=sum(x['suspicious'] for x in m),
                comparisons=sum(x['comparisons'] for x in m))
    write_json(args.sam/'comparison_summary.json',dict(schema_status='PASS',matched_frames=True,
        caveat='One assisted-seed episode; smoother boxes do not establish accuracy. Left-wrist avocado unseeded and excluded from paired aggregate.',per_camera=summary))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
