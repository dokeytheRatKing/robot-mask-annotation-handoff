"""Render review videos directly from saved RLE; never rerun or modify predictions."""
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np

from annotate import QAVideo,write_json,sha
from masks import decode

COLORS={2:(20,225,245),4:(30,115,245),10:(130,115,250),20:(55,210,65),21:(180,70,100)}


def render(image,objects,robot,title,robot_only=False):
    h,w=image.shape[:2];out=image.copy();rm=decode(robot['mask'])
    out[rm]=(out[rm]*.55+np.array([255,210,0])*.45).astype(np.uint8)
    contours,_=cv2.findContours(rm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out,contours,-1,(255,230,0),1)
    masks=[];om=np.zeros_like(rm)
    if not robot_only:
        for row in objects:
            if row['mask'] is None or not row['visible']:continue
            m=decode(row['mask']);om|=m;c=COLORS[row['object_id']]
            out[m]=(out[m]*.5+np.array(c)*.5).astype(np.uint8);masks.append(row)
        out[om&rm]=[255,0,255]
    canvas=np.concatenate([cv2.resize(image,(640,360)),cv2.resize(out,(640,360))],1)
    canvas=np.pad(canvas,((32,0),(0,0),(0,0)))
    for row in masks:
        x0,y0,x1,y1=row['bbox_xyxy'];x0=int(x0*640/w)+640;x1=int(x1*640/w)+640
        y0=int(y0*360/h)+32;y1=int(y1*360/h)+32;c=COLORS[row['object_id']]
        cv2.rectangle(canvas,(x0,y0),(x1,y1),c,1)
        text=f"{row['class_name']} {row['confidence']:.2f}"
        textwidth=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.43,1)[0][0]
        tx=min(x0,1278-textwidth);ty=max(47,y0-3)
        cv2.rectangle(canvas,(tx,ty-12),(tx+textwidth,ty+2),(15,15,15),-1)
        cv2.putText(canvas,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.43,c,1,cv2.LINE_AA)
    cv2.putText(canvas,f'{title} | RGB left | robot cyan | overlap magenta',(6,13),cv2.FONT_HERSHEY_SIMPLEX,.41,(255,255,255),1)
    unknown=[r['class_name'] for r in objects if r['visible'] is None]
    caption=f'Robot presence score {robot["confidence"]:.2f} (not mask quality)'
    if unknown:caption+=' | uninitialized: '+','.join(unknown)
    cv2.putText(canvas,caption,(6,27),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
    return canvas


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    config=json.loads((root/'run_config.json').read_text());ep=json.loads((root/'selected_episodes.json').read_text())['episodes'][0]
    selected={'head':[330,340,470,690,755,860,890,895,910,915,1050,1065],
              'left_wrist':[110,200,450,505,540,545,600,660,760,860,895,930,1070],
              'right_wrist':[180,200,210,455,505,515,625,640,1000]}
    html=['<!doctype html><meta charset="utf-8"><title>SAM2 object + robot mask pilot</title>',
       '<style>body{font-family:system-ui;background:#171b20;color:#eee;max-width:1320px;margin:2em auto}video,img{width:100%;max-width:1280px}a{color:#83caff}section{margin:2em 0}summary{cursor:pointer}</style>',
       '<h1>Object + robot segmentation mask pilot</h1>',
       '<p>Task24 / episode_002478; every 5th frame. Left: original RGB. Right: predicted masks. Cyan: robot, magenta: object/robot overlap. Scores are not calibrated mask quality. No masks were subtracted or deleted. Robot components have no asserted anatomical left/right identity.</p>',
       '<p><a href="../../docs/object_robot_mask_pilot_report_2026-09-16.md">Quality report</a></p>']
    for camera in config['cameras']:
        cache=Path(config['source_run'])/'frame_cache'/ep['episode_id']/camera
        metadata=json.loads((cache/'frame_map.json').read_text())
        folder=root/f'task_{ep["task_id"]:02d}'/ep['episode_id']
        objects=defaultdict(list)
        for r in [json.loads(x) for x in (folder/f'{camera}.objects.jsonl').read_text().splitlines()]:objects[r['frame_idx']].append(r)
        robot={r['frame_idx']:r for r in [json.loads(x) for x in (folder/f'{camera}.robot.jsonl').read_text().splitlines()]}
        video=root/'qa_masks'/f'{camera}.mp4';robot_video=root/'qa_robot_masks'/f'{camera}.mp4'
        for path in [video,robot_video]:
            if path.exists():raise FileExistsError('Choose a new visualization version instead of overwriting')
        writers=[QAVideo(video,ep['fps']/config['stride']),QAVideo(robot_video,ep['fps']/config['stride'])]
        stills=root/'review_frames'/camera;stills.mkdir(parents=True,exist_ok=True)
        contact=[]
        for local,base in enumerate(metadata):
            idx=base['frame_idx'];image=cv2.imread(str(cache/f'{local:06d}.jpg'))
            normal=render(image,objects[idx],robot[idx],f'{camera} frame={idx}')
            only=render(image,objects[idx],robot[idx],f'{camera} frame={idx}',True)
            writers[0].add(normal);writers[1].add(only)
            if idx in selected[camera] or idx%100==0 or local==len(metadata)-1:
                cv2.imwrite(str(stills/f'{idx:06d}.jpg'),normal)
                cv2.imwrite(str(stills/f'{idx:06d}.robot.jpg'),only)
            if idx%200==0 or local==len(metadata)-1:contact.append(cv2.resize(normal,(960,294)))
        for writer in writers:writer.close()
        cv2.imwrite(str(stills/'overview.jpg'),cv2.vconcat(contact))
        html.extend([f'<section><h2>{escape(camera)} — objects + robot</h2><video controls preload="metadata" src="qa_masks/{camera}.mp4"></video>',
            f'<details><summary>Robot-only overlay</summary><video controls preload="none" src="qa_robot_masks/{camera}.mp4"></video></details>',
            f'<details><summary>Timeline overview</summary><img loading="lazy" src="review_frames/{camera}/overview.jpg"></details></section>'])
    (root/'qa_index.html').write_text('\n'.join(html)+'\n')
    write_json(root/'qa_render_provenance.json',dict(script_sha256=sha(__file__),videos_sha256={
        str(p.relative_to(root)):sha(p) for group in ['qa_masks','qa_robot_masks'] for p in (root/group).glob('*.mp4')}))
    print(root/'qa_index.html')


if __name__=='__main__':main()
