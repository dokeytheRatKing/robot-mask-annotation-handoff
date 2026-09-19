"""Class-level diagnostics, not tracking or ground-truth accuracy estimates."""
import math


def geometry(previous,current,width,height):
    x0,y0,x1,y1=previous;u0,v0,u1,v1=current
    a=(x1-x0)*(y1-y0);b=(u1-u0)*(v1-v0)
    inter=max(0,min(x1,u1)-max(x0,u0))*max(0,min(y1,v1)-max(y0,v0))
    iou=inter/max(1e-9,a+b-inter)
    jump=math.hypot((u0+u1-x0-x1)/2,(v0+v1-y0-y1)/2)/math.hypot(width,height)
    ratio=max(a,b)/max(1e-9,min(a,b))
    return dict(iou=iou,center_jump_diagonal=jump,area_ratio=ratio)


def compare(previous,current,frame_idx,previous_idx,ids,width,height):
    result=[]
    for oid in ids:
        before=[d for d in previous if d['object_id']==oid]
        after=[d for d in current if d['object_id']==oid]
        record=dict(object_id=oid,frame_idx=frame_idx,previous_frame_idx=previous_idx,
                    frame_gap=frame_idx-previous_idx,previous_count=len(before),current_count=len(after),
                    association='highest_confidence_per_class_no_tracking',reasons=[])
        if before and after:
            a=max(before,key=lambda d:d['confidence']);b=max(after,key=lambda d:d['confidence'])
            record.update(geometry(a['bbox_xyxy'],b['bbox_xyxy'],width,height))
            if record['iou']<.05:record['reasons'].append('low_iou')
            if record['center_jump_diagonal']>.15:record['reasons'].append('center_jump')
            if record['area_ratio']>3:record['reasons'].append('area_jump')
        elif before:record['reasons'].append('detection_disappeared')
        elif after:record['reasons'].append('detection_reappeared')
        record['suspicious']=bool(record['reasons']);result.append(record)
    return result
