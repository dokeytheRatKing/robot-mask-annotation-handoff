"""Full engineering verification and model-only diagnostics, explicitly not GT."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import cv2
import numpy as np
from annotate import BASE,write_json,sha
from data import frames
from masks import decode,bbox
from recovery import Settings,diagnostics
from recovery_pilot import panel


def read(path):return list(map(json.loads,path.read_text().splitlines()))


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    config=json.loads((root/'run_config.json').read_text());complete=json.loads((root/'run_complete.json').read_text())
    eps=json.loads((root/'selected_episodes.json').read_text())['episodes'];mapping=json.loads((BASE/'config/task_objects.json').read_text())
    totals=Counter();summary=[]
    for ep in eps:
        for cam in config['cameras']:
            folder=root/f'task_{ep["task_id"]:02d}'/ep['episode_id'];cache=root/'frame_cache'/ep['episode_id']/cam
            meta=json.loads((cache/'frame_map.json').read_text());ids=mapping[str(ep['task_id'])]
            actual=[(idx,ts,image.shape[:2]) for idx,ts,image in frames(ep,cam,config['stride'],config['max_frames'])]
            assert [(m['frame_idx'],m['timestamp'],(m['image_height'],m['image_width'])) for m in meta]==actual
            marker=json.loads((folder/f'{cam}.complete.json').read_text())
            for name,digest in marker['files_sha256'].items():assert sha(folder/name)==digest
            rows={mode:read(folder/f'{cam}.{mode}.jsonl') for mode in ['baseline','recovery']}
            bymode={};counts=defaultdict(Counter)
            robots={r['frame_idx']:r for r in read(folder/f'{cam}.robot.jsonl')};assert len(robots)==len(meta)
            for mode,values in rows.items():
                assert len(values)==len(meta)*len(ids);bymode[mode]=defaultdict(dict)
                for r in values:
                    assert r['object_id'] not in bymode[mode][r['frame_idx']]
                    bymode[mode][r['frame_idx']][r['object_id']]=r
                previous={}
                for m in meta:
                    rs=bymode[mode][m['frame_idx']];assert set(rs)==set(ids)
                    masks={};robot=robots[m['frame_idx']]
                    rm=decode(robot['mask']) if robot['mask'] is not None else None
                    if rm is not None:assert rm.shape==(m['image_height'],m['image_width'])
                    for oid,r in rs.items():
                        assert all(r[k]==m[k] for k in m)
                        totals['object_rows']+=1
                        if r['mask'] is None:
                            assert r['visible'] is None and r['bbox_xyxy'] is None;counts[mode]['uninitialized']+=1;continue
                        binary=decode(r['mask']);assert binary.shape==(m['image_height'],m['image_width'])
                        assert int(binary.sum())==r['mask_area'] and bool(binary.any())==r['visible']
                        assert bbox(binary)==r['bbox_xyxy']
                        assert r['confidence'] is not None and 0<=r['confidence']<=1
                        masks[oid]=binary;totals['decoded_object_rles']+=1
                        counts[mode]['nonempty']+=int(binary.any());counts[mode]['empty']+=int(not binary.any())
                    flags=diagnostics(masks,previous,rm,Settings())
                    for f in flags.values():
                        counts[mode]['suspicious_object_frames']+=int(bool([x for x in f if x!='empty_mask']))
                        for flag in f:counts[mode][flag.split(':')[0]]+=1
                    previous=masks
            events=read(folder/f'{cam}.events.jsonl');detections={r['frame_idx']:r for r in read(folder/f'{cam}.detections.jsonl')}
            for e in events:
                idx=e['frame_idx'];oid=e['object_id']
                if e['event']=='initial_seed_shared':
                    x=bymode['baseline'][idx][oid];y=bymode['recovery'][idx][oid]
                    assert x['mask']==y['mask'] and x['provenance']['seed']==y['provenance']['seed']
                if e['candidate'] is not None:assert e['candidate'] in detections[idx]['detections']
                if e['event']=='terminate':
                    assert e['termination_reason'] and bymode['recovery'][idx][oid]['visible'] is False
            old=Path(config['robot_reference'])/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{cam}.robot.jsonl'
            if old.exists():assert read(old)==list(robots.values()),'Robot reference modified'
            totals['frames']+=len(meta);totals['streams']+=1
            item=dict(episode_id=ep['episode_id'],camera=cam,frames=len(meta),model_only_diagnostics={k:dict(v) for k,v in counts.items()},
                events=dict(Counter(e['event'] for e in events)),wall_seconds=marker['wall_seconds'])
            summary.append(item)
            selected={0,200,340,450,505,860,915,1000,1200}
            selected|={e['frame_idx'] for e in events if e['event']!='initial_seed_shared'}
            stilldir=root/'review_frames'/ep['episode_id']/cam;stilldir.mkdir(parents=True,exist_ok=True)
            for local,m in enumerate(meta):
                idx=m['frame_idx']
                if idx not in selected:continue
                image=cv2.imread(str(cache/f'{local:06d}.jpg'));robot=robots[idx]
                rm=decode(robot['mask']) if robot['mask'] is not None else None
                parts=[panel(image,list(bymode[mode][idx].values()),rm,f'{mode} {cam} frame={idx}') for mode in ['baseline','recovery']]
                rgb=np.pad(cv2.resize(image,(640,360)),((32,0),(0,0),(0,0)))
                cv2.putText(rgb,'RGB',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                assert cv2.imwrite(str(stilldir/f'{idx:06d}.jpg'),np.concatenate([rgb]+parts,axis=1))
    assert len(summary)==len(complete['streams'])
    write_json(root/'engineering_acceptance.json',dict(status='PASS',counts=dict(totals),streams=summary,
        quality='UNMEASURED_WITHOUT_HUMAN_GT; diagnostic changes are not IoU/accuracy improvements'))
    print(json.dumps(dict(status='PASS',counts=dict(totals)),indent=2))


if __name__=='__main__':main()
