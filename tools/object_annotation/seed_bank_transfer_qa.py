"""Render frozen transfer predictions without changing inference artifacts."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from annotate import sha, write_json
from data import frames
from seed_bank_transfer import MODES, old_rows, panel


class Video:
    def __init__(self,path,fps):
        self.path=path;self.shape=None;self.proc=None;self.fps=fps

    def add(self,image):
        if self.proc is None:
            self.shape=image.shape
            h,w,ch=self.shape
            assert ch==3 and h%2==0 and w%2==0
            self.proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n',
                '-f','rawvideo','-pixel_format','bgr24','-video_size',f'{w}x{h}',
                '-framerate',str(self.fps),'-i','pipe:0','-an','-c:v','libx264',
                '-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)
        assert image.shape==self.shape and image.dtype==np.uint8
        self.proc.stdin.write(image.tobytes())

    def close(self):
        assert self.proc is not None
        self.proc.stdin.close()
        assert self.proc.wait()==0


def labelled(image,rows,title):
    out=panel(image,rows,title)
    visible=[r for r in rows if r.get('visible') and r.get('bbox_xyxy')]
    # A separate legend avoids unreadable labels on adjacent small objects.
    legend=np.zeros((max(1,len(visible))*16,426,3),dtype=np.uint8)
    for i,r in enumerate(visible):
        oid=r['object_id'];oid=oid if isinstance(oid,int) else 'robot'
        text=f'{oid}: {r["class_name"]}'
        if r.get('confidence') is not None:text+=f' {r["confidence"]:.2f}'
        cv2.putText(legend,text,(5,12+i*16),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
        if isinstance(oid,int):
            color=tuple(map(int,np.random.default_rng(oid+7).integers(65,245,3)))
            x,y,_,_=r['bbox_xyxy'];x=min(401,max(0,int(x*426/image.shape[1])))
            y=min(261,max(38,int(y*240/image.shape[0])+26))
            cv2.putText(out,str(oid),(x,y),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    return out,legend


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();cfg=json.loads((a.root/'run_config.json').read_text())
    target=a.root/'qa_verified';target.mkdir(exist_ok=False);manifest=[]
    cv2.setNumThreads(1)
    for c in cfg['cases']:
        src=a.root/c['case_id'];dest=target/c['case_id'];dest.mkdir()
        meta=json.loads((src/'complete.json').read_text());ep=c['episode'];cam=c['camera']
        layers={}
        for mode in MODES:
            assert sha(src/f'{mode}.jsonl.gz')==meta['output_sha256'][mode]
            layers[mode]=defaultdict(list)
            with gzip.open(src/f'{mode}.jsonl.gz','rt') as f:
                for row in map(json.loads,f):layers[mode][row['frame_idx']].append(row)
        old,_=old_rows(ep,cam,c['start'],c['end'],'objects')
        robot,_=old_rows(ep,cam,c['start'],c['end'],'robot')
        # Fixed dimensions for the entire video, including fluctuating visibility.
        legend_height=max(1,len(c['object_ids']))*16
        contacts=set(np.linspace(c['start'],c['end'],5,dtype=int).tolist())|set(c['seed_frames'])
        if c['eval_frame'] is not None:contacts.add(c['eval_frame'])
        writer=Video(dest/'comparison.mp4',ep['fps']);n=0
        for idx,_,image in frames(ep,cam):
            if idx<c['start']:continue
            if idx>c['end']:break
            specs=[([],f'{c["case_id"]} {idx} RGB'),(old[idx],'Frozen production'),
                   *[(layers[m][idx],m) for m in MODES],(robot[idx],'Robot QA, unchanged')]
            cells=[]
            for rows,title in specs:
                rgb,legend=labelled(image,rows,title)
                legend=np.pad(legend,((0,legend_height-legend.shape[0]),(0,0),(0,0)))
                cells.append(np.concatenate([rgb,legend],0))
            canvas=np.concatenate([np.concatenate(cells[:3],1),np.concatenate(cells[3:],1)],0)
            writer.add(canvas);n+=1
            if idx in contacts:assert cv2.imwrite(str(dest/f'qa_{idx}.jpg'),canvas)
        writer.close();cap=cv2.VideoCapture(str(dest/'comparison.mp4'));decoded=0
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==writer.shape;decoded+=1
        cap.release();assert n==decoded==c['end']-c['start']+1
        manifest.append(dict(case_id=c['case_id'],path=str((dest/'comparison.mp4').relative_to(a.root)),
                             frames=n,shape=writer.shape,sha256=sha(dest/'comparison.mp4')))
        print(f'QA PASS {c["case_id"]} {n} frames',flush=True)
    write_json(a.root/'qa_manifest.json',dict(status='PASS',script_sha256=sha(__file__),
        supersedes='Unverified original comparison.mp4 used a legacy fixed-size writer; never use it for review.',videos=manifest))


if __name__=='__main__':main()
