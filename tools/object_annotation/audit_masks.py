#!/usr/bin/env python3
"""Independent mask schema validation and contamination/consistency diagnostics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

from annotate import BASE,sha,write_json,line
from data import frames
from masks import decode,bbox


def read(path):return [json.loads(x) for x in path.read_text().splitlines()]


def validate(row,h,w):
    if row['mask'] is None:
        assert row['mask_status']=='not_initialized'
        assert all(row[k] is None for k in ['visible','bbox_xyxy','confidence','mask_area'])
        return None
    mask=decode(row['mask'])
    assert mask.shape==(h,w)
    assert np.array_equal(mask.astype(np.uint8),coco.decode(dict(size=row['mask']['size'],counts=row['mask']['counts'].encode())))
    assert row['mask_area']==int(mask.sum())
    assert row['bbox_xyxy']==bbox(mask)
    assert row['visible']==bool(mask.any())
    assert row['mask_status']==('predicted' if mask.any() else 'predicted_empty')
    assert np.isfinite(row['confidence']) and 0<=row['confidence']<=1
    return mask


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    root=a.run;config=json.loads((root/'run_config.json').read_text())
    ep=json.loads((root/'selected_episodes.json').read_text())['episodes'][0]
    task_objects=json.loads((BASE/'config/task_objects.json').read_text())
    objects_config={x['object_id']:x['class_name'] for x in json.loads((BASE/'config/objects.json').read_text())}
    assert sha(config['checkpoint'])==config['checkpoint_sha256']
    source=Path(config['source_run']);aggregate=[];failures=[];counts=defaultdict(int)
    all_diags=[];pair_diags=[];robot_diags=[];prior_bbox_differences=[]
    for camera in config['cameras']:
        folder=root/f'task_{ep["task_id"]:02d}'/ep['episode_id']
        objects=read(folder/f'{camera}.objects.jsonl');robot=read(folder/f'{camera}.robot.jsonl')
        components=read(folder/f'{camera}.robot_components.jsonl')
        marker=json.loads((folder/f'{camera}.complete.json').read_text())
        for name,digest in marker['files_sha256'].items():assert sha(folder/name)==digest
        cache=source/'frame_cache'/ep['episode_id']/camera
        for name,digest in json.loads((folder/f'{camera}.frame_cache_hashes.json').read_text()).items():assert sha(cache/name)==digest
        bases=json.loads((cache/'frame_map.json').read_text())
        source_frames=[(idx,ts,img.shape[:2]) for idx,ts,img in frames(ep,camera,config['stride'])]
        assert [(b['frame_idx'],b['timestamp'],(b['image_height'],b['image_width'])) for b in bases]==source_frames
        byobject=defaultdict(list);bycomponent=defaultdict(list)
        for r in objects:byobject[r['frame_idx']].append(r)
        for r in components:bycomponent[r['frame_idx']].append(r)
        assert len(robot)==len(bases)==marker['sampled_images']
        expected=set(task_objects[str(ep['task_id'])])
        assert len(objects)==len(expected)*len(bases)
        assert len(components)==marker['robot_components']*len(bases)
        previous={};prev_robot=None;previous_idx=None;per=defaultdict(lambda:defaultdict(list))
        prior=read(source/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{camera}.jsonl')
        prior={(r['frame_idx'],r['object_id']):r for r in prior}
        for base,rr in zip(bases,robot):
            idx=base['frame_idx'];h,w=base['image_height'],base['image_width']
            assert set(r['object_id'] for r in byobject[idx])==expected
            assert len(byobject[idx])==len(expected)
            for row in byobject[idx]+bycomponent[idx]+[rr]:
                assert all(row[k]==base[k] for k in base)
            rm=validate(rr,h,w);union=np.zeros((h,w),bool)
            for row in bycomponent[idx]:union|=validate(row,h,w);counts['component_rows']+=1
            assert np.array_equal(rm,union),'Robot union mixed in non-robot masks or lost a component'
            counts['robot_rows']+=1
            masks={}
            for row in byobject[idx]:
                counts['object_rows']+=1;oid=row['object_id'];binary=validate(row,h,w)
                assert row['class_name']==objects_config[oid]
                if binary is None:
                    counts['unknown_object_rows']+=1;per[oid]['unknown'].append(1);continue
                counts['decoded_object_masks']+=1
                if row['visible']:counts['visible_object_rows']+=1
                masks[oid]=binary;area=int(binary.sum());shared=int((binary&rm).sum())
                entry=dict(camera=camera,frame_idx=idx,object_id=oid,class_name=row['class_name'],
                    area=area,visible=row['visible'],robot_overlap_pixels=shared,
                    object_fraction_on_robot=shared/area if area else 0.,
                    bbox_border_touch=bool(area and (binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any())),
                    suspicious_reasons=[])
                if area and shared/area>.1:entry['suspicious_reasons'].append('object_robot_overlap_gt_10pct')
                if oid in previous:
                    old=previous[oid];oldarea=int(old.sum());inter=int((old&binary).sum());un=int((old|binary).sum())
                    entry.update(previous_frame_idx=previous_idx,frame_gap=idx-previous_idx,
                        mask_iou=inter/un if un else None,area_ratio=max(area,oldarea)/min(area,oldarea) if area and oldarea else None)
                    if area and oldarea:
                        if entry['mask_iou']<.05:entry['suspicious_reasons'].append('mask_iou_lt_005')
                        if entry['area_ratio']>3:entry['suspicious_reasons'].append('area_jump_gt_3')
                    elif bool(area)!=bool(oldarea):entry['suspicious_reasons'].append('visibility_transition')
                all_diags.append(entry)
                per[oid]['rows'].append(entry)
                oldrow=prior.get((idx,oid))
                if (oldrow is None)!=(not row['visible']) or (oldrow is not None and row['bbox_xyxy']!=oldrow['bbox_xyxy']):
                    prior_bbox_differences.append(dict(camera=camera,frame_idx=idx,object_id=oid))
            for i,oid in enumerate(sorted(masks)):
                for other in sorted(masks)[i+1:]:
                    m,n=masks[oid],masks[other];small=min(int(m.sum()),int(n.sum()));inter=int((m&n).sum())
                    if inter and small and inter/small>.1:
                        pair_diags.append(dict(camera=camera,frame_idx=idx,object_id=oid,other_object_id=other,
                            shared_pixels=inter,fraction_of_smaller_mask=inter/small,
                            suspicious_reason='object_masks_overlap_gt_10pct_not_proof_of_identity_switch'))
            rd=dict(camera=camera,frame_idx=idx,mask_area=int(rm.sum()),visible=bool(rm.any()))
            if prev_robot is not None:
                union_pixels=int((rm|prev_robot).sum())
                rd['mask_iou']=float((rm&prev_robot).sum()/union_pixels) if union_pixels else None
            robot_diags.append(rd);previous=masks;prev_robot=rm;previous_idx=idx
        for oid in sorted(per):
            values=per[oid]['rows'];fractions=[v['object_fraction_on_robot'] for v in values if v['visible']]
            ious=[v['mask_iou'] for v in values if v.get('mask_iou') is not None and v.get('area_ratio') is not None]
            aggregate.append(dict(camera=camera,object_id=oid,rows=len(values)+len(per[oid]['unknown']),
                unknown=len(per[oid]['unknown']),visible=sum(v['visible'] for v in values),
                overlap_gt_10pct=sum(v['object_fraction_on_robot']>.1 for v in values),
                median_overlap=float(np.median(fractions)) if fractions else None,
                max_overlap=max(fractions) if fractions else None,
                median_mask_iou=float(np.median(ious)) if ious else None,
                temporal_jumps=sum(bool(set(v['suspicious_reasons'])&{'mask_iou_lt_005','area_jump_gt_3'}) for v in values),
                visibility_transitions=sum('visibility_transition' in v['suspicious_reasons'] for v in values)))
    for name,values in [('object_diagnostics',all_diags),('object_pair_overlaps',pair_diags),('robot_diagnostics',robot_diags)]:
        with (root/f'{name}.jsonl').open('w') as f:
            for value in values:line(f,value)
    with (root/'mask_quality.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    summary=dict(status='SCHEMA_RLE_GEOMETRY_TIMESTAMPS_PASS',counts=dict(counts),
        differences_vs_previous_bbox_run=len(prior_bbox_differences),
        object_pair_overlap_flags=len(pair_diags),quality='REQUIRES_VISUAL_REVIEW_NOT_GT_ACCEPTANCE',
        caution='Model overlap is a diagnostic, not measured true gripper contamination; robot masks may also be wrong.')
    write_json(root/'acceptance.json',summary);write_json(root/'bbox_differences.json',prior_bbox_differences)
    print(json.dumps(summary,indent=2));print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
