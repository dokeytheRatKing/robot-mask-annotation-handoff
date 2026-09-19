"""Independent verification of local causal repairs and exact untouched rows."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from annotate import PROJECT, sha, write_json
from masks import decode


def rows(path):
    with gzip.open(path,'rt') as f:
        result={}
        for r in map(json.loads,f):
            key=r['frame_idx'],r['object_id'];assert key not in result;result[key]=r
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=PROJECT/'annotations/semantic_interval_repair_20260918');a=p.parse_args()
    cfg=json.loads((a.root/'config.json').read_text());parent=Path(cfg['source'])
    pcfg=json.loads((parent/'config.json').read_text());sources={c['case_id']:c for c in pcfg['cases']}
    assert sha(parent/'config.json')==cfg['source_config_sha256']
    seedpath=a.root/'seeds/seeds.json';reviewpath=a.root/'seeds/review.json'
    review=json.loads(reviewpath.read_text());assert review['seed_sha256']==sha(seedpath)
    assert review['human_confirmed'] is False
    for r in json.loads(seedpath.read_text()):
        m=decode(r['mask']);p=r['prompt'];assert bool(m.any())==(p['status']=='visible')
        for (x,y),label in zip(p.get('points',[]),p.get('point_labels',[])):assert bool(m[y,x])==bool(label)
    modifications={};total=0;unchanged=0;unknown=0;visible=0
    for c in cfg['spec']['cases']:
        sid=c['source_case'];ep=sources[sid]['episode'];cam=sources[sid]['camera']
        ts=pq.read_table(ep['parquet'],columns=[ep['cameras'][cam]['timestamp_column']]).column(0).to_numpy()+ep['source_timestamp_start']
        fmeta=json.loads((parent/sid/'frames.json').read_text());old=rows(parent/sid/'recovery/objects.jsonl.gz')
        assert sha(parent/sid/'recovery/objects.jsonl.gz')==cfg['source_hashes'][sid]['objects']
        file=a.root/c['case_id']/'repair/objects.jsonl.gz';new=rows(file);meta=json.loads((file.parent/'complete.json').read_text())
        assert meta['objects_sha256']==sha(file) and meta['review_sha256']==sha(reviewpath)
        assert meta['seed_sha256']==sha(seedpath) and meta['config_sha256']==sha(a.root/'config.json')
        assert set(new)=={(i,oid) for i in range(c['start'],c['end']+1) for oid in sources[sid]['object_ids']}
        for i in range(c['start'],c['end']+1):
            cache=a.root/c['case_id']/'frames'/f'{i-c["start"]:05d}.jpg'
            assert sha(cache)==fmeta['jpeg_sha256'][i]
        for (i,oid),r in new.items():
            total+=1;assert r['timestamp']==float(ts[i])
            if oid!=c['object_id']:assert r==old[i,oid];unchanged+=1;continue
            modifications[sid,i,oid]=r
            for key in ('episode_id','task_id','camera','frame_idx','object_id','class_name','image_width','image_height','robot_subtraction'):
                assert r[key]==old[i,oid][key]
            event=max((p for p in c['prompts'] if p['frame_idx']<=i),key=lambda x:x['frame_idx'])
            assert r['provenance']['event_frame']==event['frame_idx'] and r['human_confirmed'] is False
            if event['status']!='visible':
                assert all(r[k] is None for k in ('mask','mask_area','bbox_xyxy','visible','confidence','seed_frame'))
                assert r['visibility']=='unknown';unknown+=1
            else:
                assert r['seed_frame']==event['frame_idx']
                m=decode(r['mask']);y,x=np.where(m);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
                assert m.shape==(r['image_height'],r['image_width']) and r['mask_area']==int(m.sum()) and r['bbox_xyxy']==box
                assert r['visible']==bool(m.any()) and math.isfinite(r['confidence']) and 0<=r['confidence']<=1
                visible+=bool(m.any())
        for layer in ('robot_source','robot_parts_source'):
            ref=json.loads((parent/sid/'recovery/complete.json').read_text())[layer];assert sha(ref['path'])==ref['sha256']
    merged_unchanged=0;merged_rows=0
    for file in (a.root/'merged').glob('*.jsonl.gz'):
        sid=file.name.removesuffix('.jsonl.gz');new=rows(file);old=rows(parent/sid/'recovery/objects.jsonl.gz')
        assert set(new)==set(old)
        for (i,oid),r in new.items():
            merged_rows+=1
            if (sid,i,oid) in modifications:assert r==modifications[sid,i,oid]
            else:assert r==old[i,oid];merged_unchanged+=1
    assert merged_rows==11780 and len(modifications)==969
    bundled=json.loads((a.root/'review_bundles.json').read_text());queue=json.loads((parent/'recovery_review_queue.json').read_text())
    members={i:r for b in bundled['bundles'] for i,r in zip(b['request_ids'],b['members'])}
    assert len(members)==len(queue) and all(members[i]==r for i,r in enumerate(queue))
    assert all(b['status']=='pending_assistant_review' for b in bundled['bundles'])
    write_json(a.root/'independent_validation.json',dict(status='PASS',semantic_acceptance=False,
        interval_rows=total,untouched_other_object_rows=unchanged,modified_target_rows=len(modifications),
        target_unknown_rows=unknown,target_positive_rows=visible,merged_rows=merged_rows,merged_unchanged_rows=merged_unchanged,
        queue_original=len(queue),queue_bundles=len(bundled['bundles']),robot_layers_unchanged=True))
    print((a.root/'independent_validation.json').read_text(),flush=True)


if __name__=='__main__':main()
