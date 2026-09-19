"""Deterministic diagnostics/seed gates; no GT and no robot-mask subtraction."""
from dataclasses import dataclass
import numpy as np
from masks import bbox


@dataclass(frozen=True)
class Settings:
    detection_threshold: float = .40
    ambiguous_margin: float = .06
    candidate_overlap: float = .70
    confirmation_iou: float = .25
    detection_period: int = 15
    search_period: int = 3
    missing_checks: int = 3
    empty_frames: int = 3
    low_iou: float = .05
    area_ratio: float = 3.
    center_jump: float = .15
    pair_overlap: float = .30
    robot_overlap: float = .50
    tiny_fraction: float = .0005


def box_iou(a,b):
    if a is None or b is None:return 0.
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter else 0.


def overlap_smaller(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    small=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
    return inter/small if small else 0.


def choose_candidates(detections,ids,cfg):
    """Abstain on same-class ambiguity or cross-class spatial collisions."""
    proposed={};rejected={}
    for oid in ids:
        items=sorted([d for d in detections if d['object_id']==oid and d['confidence']>=cfg.detection_threshold],
                     key=lambda d:d['confidence'],reverse=True)
        if not items:continue
        if len(items)>1 and items[0]['confidence']-items[1]['confidence']<cfg.ambiguous_margin:
            rejected[oid]='ambiguous_multiple_instances';continue
        proposed[oid]=items[0]
    conflicts=set()
    for i,a in proposed.items():
        for j,b in proposed.items():
            if i<j and overlap_smaller(a['bbox_xyxy'],b['bbox_xyxy'])>cfg.candidate_overlap:
                conflicts.update([i,j])
    for oid in conflicts:proposed.pop(oid);rejected[oid]='cross_class_box_collision'
    return proposed,rejected


def diagnostics(masks,previous,robot,cfg):
    result={oid:[] for oid in masks}
    for oid,m in masks.items():
        area=int(m.sum());old=previous.get(oid);b=bbox(m)
        if not area:result[oid].append('empty_mask')
        if area and b and (b[0]==0 or b[1]==0 or b[2]==m.shape[1] or b[3]==m.shape[0]):
            if area/m.size<cfg.tiny_fraction:result[oid].append('tiny_border_residue')
        if area and robot is not None and (m&robot).sum()/area>cfg.robot_overlap:
            result[oid].append('robot_overlap_warning')
        if old is not None and area and old.any():
            inter=int((m&old).sum());union=int((m|old).sum())
            if inter/union<cfg.low_iou:result[oid].append('mask_iou_drop')
            if max(area,int(old.sum()))/min(area,int(old.sum()))>cfg.area_ratio:result[oid].append('mask_area_jump')
            oldb=bbox(old)
            distance=np.linalg.norm((np.array(b[:2])+b[2:]-np.array(oldb[:2])-oldb[2:])/2)
            if distance/np.hypot(*m.shape)>cfg.center_jump:result[oid].append('bbox_center_jump')
    for i,m in masks.items():
        for j,n in masks.items():
            if i<j and min(int(m.sum()),int(n.sum())):
                if (m&n).sum()/min(int(m.sum()),int(n.sum()))>cfg.pair_overlap:
                    result[i].append(f'object_overlap:{j}');result[j].append(f'object_overlap:{i}')
    return result


def terminate_reason(flags,empty_streak,misses,presence,cfg):
    # Robot overlap alone MUST NOT delete, clip, or terminate an object.
    if misses<cfg.missing_checks:return None
    if empty_streak>=cfg.empty_frames:return 'lost_empty_no_detector_support'
    if 'tiny_border_residue' in flags:return 'suspected_out_of_view_residue'
    severe=any(f in ('mask_iou_drop','mask_area_jump','bbox_center_jump') or f.startswith('object_overlap:') for f in flags)
    if severe and presence<.5:return 'lost_unreliable_no_detector_support'
    return None
