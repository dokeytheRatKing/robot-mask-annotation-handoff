"""Frozen DINOv2 appearance lookup; no optimizer, training, or claimed GT."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from annotate import BASE, PROJECT, sha, write_json
from data import frames


def crop(image, box):
    h,w=image.shape[:2]
    x0,y0,x1,y1=box
    pad=.04*max(x1-x0,y1-y0)
    x0,y0=max(0,int(x0-pad)),max(0,int(y0-pad))
    x1,y1=min(w,int(x1+pad+1)),min(h,int(y1+pad+1))
    if x1<=x0 or y1<=y0: raise ValueError('Empty appearance crop')
    return image[y0:y1,x0:x1]


class Appearance:
    def __init__(self,gallery=None,project=PROJECT):
        project=Path(project)
        self.model=torch.hub.load(str(project/'third_party/dinov2'),'dinov2_vits14',
                                  source='local',pretrained=False).eval().cuda()
        self.model.load_state_dict(torch.load(project/'models/dinov2/dinov2_vits14_pretrain.pth',
                                             map_location='cpu',weights_only=True))
        self.model.requires_grad_(False)
        if gallery:
            data=np.load(gallery,allow_pickle=False)
            self.features=torch.from_numpy(data['features']).cuda()
            self.ids=data['ids']

    @torch.inference_mode()
    def embed(self,crops):
        output=[]
        for start in range(0,len(crops),32):
            rgb=np.stack([cv2.cvtColor(cv2.resize(c,(224,224)),cv2.COLOR_BGR2RGB)
                          for c in crops[start:start+32]])
            x=torch.from_numpy(rgb).cuda().permute(0,3,1,2).float()/255
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            with torch.autocast('cuda',dtype=torch.bfloat16): y=self.model(x)
            output.append(F.normalize(y.float(),dim=-1))
        return torch.cat(output) if output else torch.empty(0,384,device='cuda')

    def check(self,image,detections):
        if not detections:return []
        x=self.embed([crop(image,d['bbox_xyxy']) for d in detections])
        with torch.autocast('cuda',enabled=False):
            sims=(x.float()@self.features.float().T).cpu().numpy()
        result=[]
        for d,similarities in zip(detections,sims):
            oid=d['object_id']
            compatible={17,22,23} if oid in {17,22,23} else {oid}
            positive=np.isin(self.ids,list(compatible))
            if not positive.any():
                result.append(d|dict(appearance_status='no_reference',appearance_accepted=False));continue
            pos=float(similarities[positive].max());neg=float(similarities[~positive].max())
            winner=int(self.ids[similarities.argmax()])
            result.append(d|dict(appearance_similarity=pos,appearance_competitor_similarity=neg,
                appearance_margin=pos-neg,appearance_nearest_class=winner,
                appearance_accepted=pos>=.50 and pos-neg>=.05,
                appearance_status='frozen_exemplar_lookup_not_calibrated_probability'))
        return result


def build(args):
    root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    episodes={e['episode_id']:e for e in json.loads((root/'episodes.json').read_text())['episodes']}
    specs=json.loads((BASE/'config/appearance_seeds.json').read_text())['seeds']
    images={};crops=[];meta=[]
    for spec in specs:
        key=(spec['episode'],'head',0)
        if key not in images:images[key]=next(frames(episodes[key[0]],key[1],max_frames=1))[2]
        im=images[key];h,w=im.shape[:2]
        box=np.asarray(spec['box'])*[w/426,h/240,w/426,h/240]
        crops.append(crop(im,box));meta.append(spec|dict(camera='head',frame_idx=0,
            source='assistant_visual_crop',box_original=box.tolist()))
    # User-reviewed masks supply multiple actual camera views. Select uniformly
    # across each class/camera, not only the earliest almost identical frames.
    groups=defaultdict(list)
    for path in sorted((root/'reviewed_reference').rglob('*.jsonl')):
        for row in map(json.loads,path.read_text().splitlines()):
            if row['visible'] and row.get('mask_area',0)>300:
                groups[(row['object_id'],row['camera'])].append(row)
    selected=[]
    for rows in groups.values():
        selected.extend(rows[i] for i in np.linspace(0,len(rows)-1,min(12,len(rows)),dtype=int))
    by_stream=defaultdict(list)
    for r in selected:by_stream[(r['episode_id'],r['camera'])].append(r)
    for (ep,cam),rows in by_stream.items():
        wanted=defaultdict(list)
        for row in rows:wanted[row['frame_idx']].append(row)
        for idx,_,im in frames(episodes[ep],cam):
            if idx not in wanted:continue
            for row in wanted[idx]:
                crops.append(crop(im,row['bbox_xyxy']));meta.append(dict(episode=ep,camera=cam,frame_idx=idx,
                    object_id=row['object_id'],box_original=row['bbox_xyxy'],source='user_confirmed_mask_bbox'))
    # Explicit robot/background negatives from RGB, never mask subtraction.
    for ep,cam,idx,box in [('episode_001684','left_wrist',0,[0,272,220,360]),
                          ('episode_001684','left_wrist',0,[469,275,640,360]),
                          ('episode_002577','right_wrist',60,[465,270,640,360])]:
        im=next(im for j,_,im in frames(episodes[ep],cam) if j==idx)
        crops.append(crop(im,box));meta.append(dict(episode=ep,camera=cam,frame_idx=idx,object_id=-1,
                                                   box_original=box,source='assistant_robot_negative'))
    model=Appearance();features=model.embed(crops).cpu().numpy()
    np.savez_compressed(out/'gallery.npz',features=features,ids=np.array([r['object_id'] for r in meta]))
    tiles=[]
    for i,(im,entry) in enumerate(zip(crops,meta)):
        tile=cv2.resize(im,(112,112));tile=np.pad(tile,((20,0),(0,0),(0,0)))
        cv2.putText(tile,f'{i}:id{entry["object_id"]}',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        tiles.append(tile)
    for page in range(0,len(tiles),64):
        items=tiles[page:page+64]
        while len(items)%8:items.append(np.zeros_like(items[0]))
        sheet=np.concatenate([np.concatenate(items[j:j+8],1) for j in range(0,len(items),8)],0)
        cv2.imwrite(str(out/f'gallery_{page//64:02d}.jpg'),sheet)
    write_json(out/'gallery.json',dict(examples=meta,sha256=sha(out/'gallery.npz'),
        checkpoint_sha256=sha(PROJECT/'models/dinov2/dinov2_vits14_pretrain.pth'),
        role='Frozen feature similarity lookup, no fitting or quality ground truth'))
    print(json.dumps(dict(examples=len(meta),ids=sorted(set(r['object_id'] for r in meta)))),flush=True)


def probe(args):
    model=Appearance(args.gallery)
    episodes={e['episode_id']:e for e in json.loads((args.root/'episodes.json').read_text())['episodes']}
    rows=json.loads(args.probe.read_text());report=[]
    for r in rows:
        im=next(im for idx,_,im in frames(episodes[r['episode']],r['camera']) if idx==r['frame_idx'])
        checked=model.check(im,[d for d in r['detections'] if d['object_id']<1000])
        report.append({k:v for k,v in r.items() if k!='detections'}|dict(detections=checked))
    write_json(args.output,report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build','probe']);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--gallery',type=Path);p.add_argument('--probe',type=Path)
    args=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    build(args) if args.command=='build' else probe(args)


if __name__=='__main__':main()
