"""Anatomical robot masks from registered visual seeds, with explicit unknowns.

Only reference initialization uses the reviewed flange line. Subsequent masks
are propagated by SAM2; image left/right is never used to relabel tracks.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from annotate import BASE, PROJECT, sha, write_json
from masks import decode, encode, record

PARTS = {1101: 'left_arm', 1102: 'right_arm', 1103: 'left_ee', 1104: 'right_ee', 1100: 'robot_unknown'}
COLORS = {1101: (255, 145, 60), 1102: (60, 210, 90), 1103: (230, 70, 230), 1104: (30, 205, 255), 1100: (150, 150, 150)}


def flange_split(binary, boundary, arm_point, band=4):
    a,b=np.asarray(boundary,float)
    yy,xx=np.indices(binary.shape)
    distance=((xx-a[0])*(b[1]-a[1])-(yy-a[1])*(b[0]-a[0]))/np.linalg.norm(b-a)
    q=np.asarray(arm_point)-a
    sign=np.sign(q[0]*(b[1]-a[1])-q[1]*(b[0]-a[0]))
    return binary & (distance*sign>band), binary & (distance*sign < -band), binary & (np.abs(distance)<=band)


def resolve_conflicts(masks, shape):
    """Within-robot conflicts remain visible but unknown; never touch objects."""
    known=[masks.get(k,np.zeros(shape,bool)).astype(bool) for k in PARTS if k!=1100]
    overlap=np.sum(known,axis=0)>1
    unknown=masks.get(1100,np.zeros(shape,bool)) | overlap
    result={k:masks.get(k,np.zeros(shape,bool)) & ~unknown for k in PARTS if k!=1100}
    result[1100]=unknown
    return result


def anatomy_guard(masks, camera, verified_sides, shape):
    """An unseen flange cannot justify assigning new forearm pixels to EE."""
    guarded={k:m.copy() for k,m in masks.items()};reasons={}
    if camera=='head':
        for side,arm,ee in [('left',1101,1103),('right',1102,1104)]:
            if side in verified_sides:continue
            for k in [arm,ee]:
                m=guarded.get(k,np.zeros(shape,bool))
                if m.any():
                    guarded[1100]=guarded.get(1100,np.zeros(shape,bool))|m
                    guarded[k]=np.zeros(shape,bool)
                    reasons[k]='flange_not_observed_robot_side_known_part_boundary_unresolved'
    return resolve_conflicts(guarded,shape),reasons


def overlay(image, masks, title):
    out=image.copy()
    for k,m in masks.items():
        color=np.asarray(COLORS[k]);out[m]=(out[m]*.5+color*.5).astype(np.uint8)
    out=np.pad(out,((35,0),(0,0),(0,0)))
    cv2.putText(out,title,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    return out


class RobotSeeds:
    def __init__(self, directory):
        self.directory=Path(directory)
        self.spec=json.loads((self.directory/'references.json').read_text())
        self.sift=cv2.SIFT_create(nfeatures=1800)
        self.references=[]
        for entry in self.spec['references']:
            image=cv2.imread(str(self.directory/entry['image']))
            union=np.zeros(image.shape[:2],bool)
            for r in entry['masks'].values():union|=decode(r)
            # Restrict descriptors to the actual robot, not the scene objects.
            support=cv2.erode(union.astype(np.uint8)*255,np.ones((5,5),np.uint8))
            kp,desc=self.sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),support)
            self.references.append((entry,kp,desc))

    def locate(self,image,camera):
        h,w=image.shape[:2];masks={};details=[]
        kp,desc=self.sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None)
        if desc is None:return masks,details
        for entry,rkp,rdesc in self.references:
            if entry['camera']!=camera or rdesc is None:continue
            pairs=cv2.BFMatcher().knnMatch(rdesc,desc,k=2)
            matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.70*b.distance]
            if len(matches)<7:continue
            src=np.float32([rkp[m.queryIdx].pt for m in matches])
            dst=np.float32([kp[m.trainIdx].pt for m in matches])
            # Similarity transform is intentionally conservative at initialization;
            # unrestricted homographies can map a gripper onto a scene object.
            transform,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
            if transform is None:continue
            n=int(inliers.sum());ratio=n/len(matches);scale=np.linalg.norm(transform[:,0])
            if n<7 or ratio<.50 or not .65<scale<1.55:continue
            if camera=='head':
                # Only transfer near the reviewed initialization posture. A
                # mirrored/crossed-arm match is not evidence of anatomical side.
                center=src.mean(axis=0)
                moved=transform[:,:2]@center+transform[:,2]
                if abs(moved[0]-center[0])>.22*w or transform[0,0]<.6*scale:continue
            selected=inliers[:,0].astype(bool)
            extent=np.ptp(src[selected],axis=0)
            if np.linalg.norm(extent)<35:continue
            entry_masks={int(k):cv2.warpAffine(decode(v).astype(np.uint8),transform,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
                         for k,v in entry['masks'].items()}
            if not all(m.any() for k,m in entry_masks.items() if k!=1100):continue
            for k,m in entry_masks.items():masks[k]=masks.get(k,np.zeros((h,w),bool))|m
            details.append(dict(reference=entry['name'],part_ids=list(entry_masks),inliers=n,matches=len(matches),
                                inlier_ratio=ratio,scale=scale,transform=transform.tolist(),
                                method='foreground_SIFT_RANSAC_registered_visual_seed',
                                flange_boundary_status='approximate_visual_seed_not_human_GT'))
        return resolve_conflicts(masks,(h,w)),details


def build(args):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=json.loads((BASE/'config/robot_parts.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt'),device='cuda')
    predictor=SAM2ImagePredictor(model);refs=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for camera in ['head','left_wrist','right_wrist']:
            image=cv2.imread(str(args.images/f'{cfg["reference_episode"]}_{camera}_000000.png'));assert image is not None
            default_image=image;whole={}
            for number,spec in enumerate(cfg[camera]):
                image=cv2.imread(str(PROJECT/spec['image'])) if spec.get('image') else default_image
                assert image is not None
                predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                # Query fingers separately in wrist views: one disconnected-mask
                # query may select the basket between fingers instead of both.
                if camera!='head':
                    binary=np.zeros(image.shape[:2],bool)
                    for points in [spec['points'][:2],spec['points'][2:]]:
                        other=spec['points'][2:] if points==spec['points'][:2] else spec['points'][:2]
                        m,s,_=predictor.predict(point_coords=np.array(points+other),point_labels=np.array([1,1,0,0]),multimask_output=True)
                        valid=[i for i,x in enumerate(m) if 150<x.sum()<image.shape[0]*image.shape[1]*.25]
                        if valid:binary|=m[max(valid,key=lambda i:s[i])].astype(bool)
                    parts={spec['part_ids'][0]:binary}
                else:
                    m,s,_=predictor.predict(point_coords=np.array(spec['points']),point_labels=np.array(spec['labels']),
                                             box=np.array(spec['box']),multimask_output=False)
                    binary=m[0].astype(bool)
                    if 'flange_boundary' in spec:
                        arm,ee,unknown=flange_split(binary,spec['flange_boundary'],spec['arm_point'])
                        parts={spec['part_ids'][0]:arm,spec['part_ids'][1]:ee,1100:unknown}
                    else:parts={spec['part_ids'][0]:binary}
                name=f'{camera}_{spec["side"]}_{number}';path=f'{name}.png';cv2.imwrite(str(args.output/path),image)
                if not spec.get('image'):
                    for k,m in parts.items():whole[k]=whole.get(k,np.zeros(image.shape[:2],bool))|m
                refs.append(dict(name=name,camera=camera,image=path,image_sha256=sha(args.output/path),
                                 masks={str(k):encode(v) for k,v in parts.items()},seed=spec))
                cv2.imwrite(str(args.output/f'{name}_seed_overlay.jpg'),overlay(image,parts,name))
            cv2.imwrite(str(args.output/f'{camera}_seed_overlay.jpg'),overlay(default_image,whole,'RGB-reviewed reference: arm / EE / uncertain boundary'))
    write_json(args.output/'references.json',dict(config=cfg,references=refs,accepted_ground_truth=False))
    print(json.dumps(dict(references=len(refs),output=str(args.output))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build']);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();build(args)


if __name__=='__main__':main()
