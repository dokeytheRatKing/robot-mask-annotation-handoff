#!/usr/bin/env python3
"""Validate outputs and aggregate detection coverage, sampled-frame stability and speed."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from annotate import sha,write_json
from diagnostics import compare


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def summary(root):
    config=json.loads((root/'run_config.json').read_text())
    manifest=json.loads((root/'selected_episodes.json').read_text())['episodes']
    objects={x['object_id']:x for x in json.loads(Path(config['objects']).read_text())}
    mapping=json.loads(Path(config['task_objects']).read_text())
    aggregates={};timings=[];totalframes=0;totaldet=0;expected=0
    for ep in manifest:
        folder=root/f'task_{ep["task_id"]:02d}'/ep['episode_id']
        ids=mapping[str(ep['task_id'])]
        for cam in ('head','left_wrist','right_wrist'):
            expected+=1;marker=json.loads((folder/f'{cam}.complete.json').read_text());timings.append(marker)
            for name,digest in marker['files_sha256'].items():assert sha(folder/name)==digest
            ledger=rows(folder/f'{cam}.frames.jsonl');detections=rows(folder/f'{cam}.jsonl')
            candidate_rows=rows(folder/f'{cam}.candidates.jsonl')
            assert len(ledger)==marker['processed_frames']
            idx=[f['frame_idx'] for f in ledger]
            assert idx==list(range(0,(idx[-1]+1),config['stride']))
            if not config['max_frames'] and 'frames' in ep:
                assert len(ledger)==len(range(0,ep['frames'],config['stride']))
            byframe=defaultdict(list)
            for d in detections:
                assert d['episode_id']==ep['episode_id'] and d['task_id']==ep['task_id']
                assert d['camera']==cam and d['object_id'] in ids and d['frame_idx'] in idx
                assert d['class_name']==objects[d['object_id']]['class_name']
                assert config['threshold']<=d['confidence']<=1
                x0,y0,x1,y1=d['bbox_xyxy']
                assert 0<=x0<x1<=d['image_width'] and 0<=y0<y1<=d['image_height']
                assert np.isfinite(d['bbox_xyxy']).all()
                byframe[d['frame_idx']].append(d)
            for f in ledger:
                ds=byframe[f['frame_idx']]
                assert f['detection_count']==len(ds)
                assert f['absent_object_ids']==[oid for oid in ids if not any(d['object_id']==oid for d in ds)]
                for d in ds:
                    assert all(d[k]==f[k] for k in ('timestamp','image_width','image_height'))
            totalframes+=len(ledger);totaldet+=len(detections)
            allc=defaultdict(list)
            for d in candidate_rows:allc[d['frame_idx']].append(d)
            for threshold in [.2,.25,.3,.35,.4]:
                previous=[];previous_idx=None
                for f in ledger:
                    current=[d for d in allc[f['frame_idx']] if d['confidence']>=threshold]
                    diags=[] if previous_idx is None else compare(previous,current,f['frame_idx'],previous_idx,
                                                 ids,f['image_width'],f['image_height'])
                    dm={d['object_id']:d for d in diags}
                    for oid in ids:
                        key=(ep['task_id'],cam,oid,threshold)
                        a=aggregates.setdefault(key,dict(task_id=ep['task_id'],camera=cam,object_id=oid,
                            class_name=objects[oid]['class_name'],threshold=threshold,frames=0,detected_frames=0,
                            multi_detection_frames=0,transitions=0,suspicious=0,disappeared=0,reappeared=0,
                            ious=[],centers=[],areas=[],scores=[]))
                        ds=[d for d in current if d['object_id']==oid]
                        a['frames']+=1;a['detected_frames']+=bool(ds);a['multi_detection_frames']+=len(ds)>1
                        a['scores'].extend(d['confidence'] for d in ds)
                        if oid in dm:
                            d=dm[oid];a['transitions']+=1;a['suspicious']+=d['suspicious']
                            a['disappeared']+='detection_disappeared' in d['reasons']
                            a['reappeared']+='detection_reappeared' in d['reasons']
                            if 'iou' in d:
                                a['ious'].append(d['iou']);a['centers'].append(d['center_jump_diagonal']);a['areas'].append(d['area_ratio'])
                    previous=current;previous_idx=f['frame_idx']
    data=[]
    for a in aggregates.values():
        a['detection_fraction']=a['detected_frames']/a['frames']
        a['suspicious_fraction']=a['suspicious']/max(1,a['transitions'])
        for key in ['ious','centers','areas','scores']:
            v=a.pop(key)
            a[key+'_median']=float(np.median(v)) if v else None
            a[key+'_p95']=float(np.quantile(v,.95)) if v else None
        data.append(a)
    data.sort(key=lambda d:(d['threshold'],d['task_id'],d['camera'],d['object_id']))
    wall=sum(t['wall_seconds'] for t in timings);model=sum(t['model_seconds'] for t in timings)
    report=dict(status='SCHEMA_AND_COMPLETENESS_PASS',episode_count=len(manifest),camera_streams=expected,
        sampled_images=totalframes,detections=totaldet,stream_wall_seconds=wall,
        images_per_second=totalframes/wall,model_images_per_second=totalframes/model,
        peak_allocated_gib=max(t['peak_allocated_gib'] for t in timings),
        peak_reserved_gib=max(t['peak_reserved_gib'] for t in timings),
        qa_videos=len(list((root/'qa').rglob('*.mp4'))),
        caveats=['Detection fraction is NOT recall: visibility is not labeled.',
                 ('Diagnostics compare consecutive raw frames within the measured window.' if config['stride']==1 else
                  'Diagnostics compare adjacent sampled frames, not all raw consecutive frames.'),
                 'Highest confidence per class can switch instances; no identity tracking.',
                 'Suspicious does not mean incorrect; occlusion and camera motion can be legitimate.'],
        per_camera={cam:dict(images=sum(t['processed_frames'] for t in timings if t['camera']==cam),
                    wall_seconds=sum(t['wall_seconds'] for t in timings if t['camera']==cam))
                    for cam in ['head','left_wrist','right_wrist']})
    write_json(root/'summary.json',report)
    with (root/'object_camera_threshold_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    reporttext=['# GroundingDINO pilot: measured results','',json.dumps(report,indent=2),'',
       'Detection fraction below is observation coverage, not labeled precision/recall.','',
       '| task | camera | object | frames detected / processed | median IoU | suspicious / comparisons |',
       '| --- | --- | --- | --- | --- | --- |']
    for d in data:
        if abs(d['threshold']-config['threshold'])>1e-9:continue
        reporttext.append(f"| {d['task_id']} | {d['camera']} | {d['class_name']} | {d['detected_frames']}/{d['frames']} | {d['ious_median']} | {d['suspicious']}/{d['transitions']} |")
    (root/'measured_report.md').write_text('\n'.join(reporttext)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    summary(p.parse_args().output)
