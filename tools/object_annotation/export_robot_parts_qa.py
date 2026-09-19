"""Validate dense robot layers and export bounded, downloadable QA artifacts."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import itertools
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from annotate import write_json
from masks import decode
from robot_parts import PARTS, COLORS
from full_segmentation import render


def rows(path):
    with gzip.open(path,'rt') as f:
        yield from map(json.loads,f)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--video-episode',default='episode_001684');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    eps={e['episode_id']:e for e in json.loads((args.root/'episodes.json').read_text())['episodes']}
    summaries=[];by_camera=defaultdict(Counter);tiles=[]
    for path in sorted(args.root.glob('task_*/episode_*/*/attempt_*/complete.json')):
        result=json.loads(path.read_text())
        if 'robot_parts' not in result['files']:continue
        ep=eps[result['episode_id']];camera=result['camera'];n=result['frames']
        objects=rows(args.root/result['files']['objects']);parts=rows(args.root/result['files']['robot_parts'])
        union=rows(args.root/result['files']['robot']);counts=Counter();total=0
        cap=cv2.VideoCapture(ep['cameras'][camera]['video']);video=None
        selected=set(np.linspace(0,n-1,4,dtype=int));stem=f'{ep["episode_id"]}_{camera}'
        if ep['episode_id']==args.video_episode:
            video=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
                '-s','1280x390','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/f'{stem}.mp4')],stdin=subprocess.PIPE)
        for idx in range(n):
            rr=list(itertools.islice(parts,5));oo=list(itertools.islice(objects,len(result['object_ids'])))
            u=next(union);assert len(rr)==5 and len(oo)==len(result['object_ids'])
            assert [r['part_id'] for r in rr]==list(PARTS)
            assert all(r['frame_idx']==idx for r in rr+oo+[u])
            decoded=[decode(r['mask']) for r in rr]
            assert np.array_equal(np.logical_or.reduce(decoded),decode(u['mask']))
            assert (np.sum(decoded,axis=0)<=1).all()
            for r,m in zip(rr,decoded):
                counts[r['class_name']+'_visible']+=int(m.any())
                counts[r['class_name']+'_uninitialized']+=int(r['visible'] is None)
            if any(m.any() for m in decoded[:4]):counts['some_named_part_visible']+=1
            if any(m.any() for m in decoded):counts['some_robot_visible']+=1
            total+=1
            if video is not None or idx in selected:
                cap.set(cv2.CAP_PROP_POS_FRAMES,idx) if video is None else None
                ok,image=cap.read();assert ok
                preview=render(image,oo,u,f'{stem} frame {idx}; robot parts are unconfirmed automatic masks',rr)
                if video is not None:video.stdin.write(preview.tobytes())
                if idx in selected:
                    cv2.imwrite(str(args.output/f'{stem}_{idx:06d}.jpg'),preview)
                    if idx==0:tiles.append(cv2.resize(preview,(960,292)))
        cap.release()
        assert next(parts,None) is None and next(union,None) is None and next(objects,None) is None
        if video is not None:
            video.stdin.close();assert video.wait()==0
        by_camera[camera].update(counts);by_camera[camera]['frames']+=total
        summaries.append(dict(episode=ep['episode_id'],camera=camera,frames=total,counts=dict(counts),
                              seconds=result['wall_seconds'],peak_allocated_gib=result['peak_allocated_gib']))
        print(stem,total,flush=True)
    for start in range(0,len(tiles),6):
        page=tiles[start:start+6]
        cv2.imwrite(str(args.output/f'overview_{start//6}.jpg'),np.concatenate(page,axis=0))
    report=dict(structural_status='PASS' if summaries else 'NO_OUTPUT',streams=summaries,
        frames=sum(r['frames'] for r in summaries),cameras={k:dict(v) for k,v in by_camera.items()},
        quality='Coverage is not IoU or anatomical correctness. No human robot-mask ground truth exists.',
        semantics='Flange belongs to arm. EE includes gripper body and fingers. Uncertain parts remain unknown. Objects are never subtracted.')
    write_json(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='streams'},indent=2))


if __name__=='__main__':main()
