"""Independent geometry, lineage, temporal-key and reference-integrity checks."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from annotate import PROJECT, BASE, sha, write_json
from masks import decode


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=PROJECT/'annotations/full_episode_reentry_20260918')
    a=p.parse_args();root=a.root;cfg=json.loads((root/'config.json').read_text());counts=Counter();summaries=[]
    for name,digest in cfg['code'].items():assert sha(BASE/name)==digest
    assert sha(PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt')==cfg['models']['sam2']
    bank=json.loads((PROJECT/'annotations/identity_seed_bank_20260918_curated/bank.json').read_text())
    entries={r['exemplar_id']:r for r in bank['entries']}
    seed_path=root/'seeds_r2/seeds.json';seeds=json.loads(seed_path.read_text())
    review_path=root/'seeds_r2/review.json';review=json.loads(review_path.read_text())
    assert review['seed_sha256']==sha(seed_path) and review['human_confirmed'] is False
    for seed in seeds:
        m=decode(seed['mask']);assert seed['human_confirmed'] is False
        assert bool(m.any())==(seed['prompt']['status']=='visible')
        for spec in [seed['prompt'],*seed['prompt'].get('components',[])]:
            for xy,label in zip(spec.get('points',[]),spec.get('point_labels',[])):
                if label==1:assert m[xy[1],xy[0]],(seed['case_id'],seed['frame_idx'],xy)
    for c in cfg['cases']:
        folder=root/c['case_id'];ep=c['episode'];n=ep['frames'];cam=c['camera'];fm=json.loads((folder/'frames.json').read_text())
        ts=pq.read_table(ep['parquet'],columns=[ep['cameras'][cam]['timestamp_column']]).column(0).to_numpy()+ep['source_timestamp_start']
        assert np.array_equal(np.array(fm['timestamps']),ts)
        assert sha(ep['cameras'][cam]['video'])==fm['source_video_sha256']
        for idx,digest in enumerate(fm['jpeg_sha256']):assert sha(folder/'frames'/f'{idx:05d}.jpg')==digest
        for row in json.loads((folder/'initial_candidates.json').read_text()):
            r=row['retrieval']
            refs=[*r.get('nearest_exemplar_ids',[])]
            if r.get('competitor_exemplar_id'):refs.append(r['competitor_exemplar_id'])
            for ref in refs:assert entries[ref]['episode_id']!=ep['episode_id']
        for mode in ('baseline','recovery'):
            out=folder/mode;meta=json.loads((out/'complete.json').read_text());path=out/'objects.jsonl.gz'
            assert meta['config_sha256']==sha(root/'config.json')
            assert meta['initial_sha256']==sha(folder/'initial_candidates.json')
            assert meta['prior_sha256']==sha(folder/'prior_seeds.json')
            if mode=='recovery':assert meta['review_sha256']==sha(review_path)
            for layer in ('robot_source','robot_parts_source'):
                assert sha(meta[layer]['path'])==meta[layer]['sha256']
            assert sha(path)==meta['objects_sha256'] and sha(out/'events.json')==meta['events_sha256']
            events=json.loads((out/'events.json').read_text());byid={oid:[] for oid in c['object_ids']}
            for event in events:
                assert 0<=event['frame_idx']<n;byid[event['object_id']].append(event)
            stats=Counter();seen=set()
            with gzip.open(path,'rt') as f:
                for row in map(json.loads,f):
                    idx=row['frame_idx'];oid=row['object_id'];key=(idx,oid)
                    assert key not in seen;seen.add(key)
                    assert row['episode_id']==ep['episode_id'] and row['task_id']==ep['task_id'] and row['camera']==cam
                    assert row['timestamp']==float(ts[idx]) and math.isfinite(row['timestamp'])
                    assert row['human_confirmed'] is False and row['robot_subtraction'] is False
                    assert row['image_height']==fm['shape'][0] and row['image_width']==fm['shape'][1]
                    causal=[e for e in byid[oid] if e['frame_idx']<=idx]
                    event=causal[-1] if causal else None
                    if row['mask'] is None:
                        assert event is None or event['action']=='clear_unknown'
                        assert all(row[k] is None for k in ('visible','confidence','bbox_xyxy','mask_area','provenance'))
                        stats['unknown_unseeded']+=1
                    else:
                        assert event and event['action']=='seed' and event['frame_idx']==row['seed_frame']
                        assert row['provenance']['frame_idx']==row['seed_frame']<=idx
                        m=decode(row['mask']);assert m.shape==tuple(fm['shape'][:2])
                        y,x=np.where(m);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
                        assert row['bbox_xyxy']==box and row['mask_area']==int(m.sum()) and row['visible']==bool(m.any())
                        assert math.isfinite(row['confidence']) and 0<=row['confidence']<=1
                        stats['predicted_visible' if m.any() else 'unknown_empty']+=1
                    stats.update('flag/'+s for s in row['suspicious_flags']);stats['rows']+=1
            assert seen=={(idx,oid) for idx in range(n) for oid in c['object_ids']}
            counts['rows']+=stats['rows'];counts['streams']+=1
            summaries.append(dict(case_id=c['case_id'],mode=mode,seconds=meta['seconds'],peak_allocated_gib=meta['peak_allocated_gib'],
                frames=n,seed_events=sum(e['action']=='seed' for e in events),clear_events=sum(e['action']=='clear_unknown' for e in events),**dict(stats)))
    write_json(root/'independent_validation.json',dict(status='PASS',semantic_acceptance=False,counts=dict(counts),streams=summaries))
    print(json.dumps(dict(status='PASS',counts=dict(counts))))


if __name__=='__main__':main()
