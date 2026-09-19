"""Portable evidence and reviewer contact sheets for local semantic repairs."""
import argparse
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from annotate import BASE,PROJECT,sha,write_json
from full_episode_reentry import rows_at
from semantic_interval_repair import DEFAULT,PARENT,load,image_at
from seed_bank_transfer_qa import labelled


def contacts(a):
    cfg=load(a.root);out=a.root/'checks';out.mkdir(exist_ok=False)
    for c in cfg['spec']['cases']:
        old=rows_at(PARENT/c['source_case']/'recovery/objects.jsonl.gz')
        new=rows_at(a.root/c['case_id']/'repair/objects.jsonl.gz');tiles=[]
        for idx in c['checks']:
            image=image_at(c,idx);cells=[]
            for rows,title in [([],f'{c["case_id"]} RGB {idx}'),(old[idx],'Previous recovery'),(new[idx],'Local repair')]:
                rr=[r for r in rows if r['object_id']==c['object_id']]
                rgb,legend=labelled(image,rr,title);cells.append(np.concatenate([rgb,legend],0))
            tiles.append(np.concatenate(cells,1))
        for start in range(0,len(tiles),4):
            cv2.imwrite(str(out/f'{c["case_id"]}_{start//4}.jpg'),np.concatenate(tiles[start:start+4],0))
    print('CONTACTS',len(list(out.glob('*.jpg'))),flush=True)


def package(a):
    validation=json.loads((a.root/'validation.json').read_text());assert validation['status']=='PASS'
    independent=json.loads((a.root/'independent_validation.json').read_text());assert independent['status']=='PASS'
    assert json.loads((a.root/'final_r3/validation.json').read_text())['status']=='PASS'
    target=PROJECT/'deliverables/astribot_semantic_interval_repair_20260918.zip';assert not target.exists()
    files=[p for p in a.root.rglob('*') if p.is_file() and 'frames' not in p.relative_to(a.root).parts]
    files += [BASE/n for n in ('semantic_interval_repair.py','semantic_interval_artifacts.py','validate_semantic_interval.py','probe_semantic_interval_bank.py','refine_semantic_boundaries.py','finalize_semantic_seams.py','test_semantic_interval.py','config/semantic_interval_repair_20260918.json','config/semantic_boundary_review_20260918.json')]
    files += [PROJECT/'docs/semantic_interval_repair_report_20260918.md']
    files += list((PROJECT/'runs').glob('semantic_interval_*_20260918.log'))
    manifest=[dict(path=str(p.relative_to(PROJECT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)]
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(PROJECT)))
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    write_json(a.root/'delivery.json',dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(files)+1,crc='PASS'))
    print((a.root/'delivery.json').read_text(),flush=True)


def finalcheck(a):
    cfg=load(a.root);folder=a.root/'boundary_r2';meta=json.loads((folder/'validation.json').read_text())
    assert meta['status']=='PASS' and meta['semantic_acceptance'] is False
    assert meta['code_sha256']==sha(BASE/'refine_semantic_boundaries.py')
    assert meta['review_sha256']==sha(BASE/'config/semantic_boundary_review_20260918.json')
    changed={(i,2) for op in meta['review']['overrides'] for i in range(op['start'],op['end']+1)}
    assert len(changed)==12
    new=rows_at(folder/'episode_002503_right_wrist.jsonl.gz')
    old=rows_at(a.root/'merged/episode_002503_right_wrist.jsonl.gz');seen=set();counts={'unknown':0,'visible':0,'empty':0}
    assert new.keys()==old.keys()
    for i,rr in new.items():
        assert len(rr)==5
        for r in rr:
            key=i,r['object_id'];assert key not in seen;seen.add(key)
            before=next(x for x in old[i] if x['object_id']==r['object_id'])
            if key in changed:
                assert all(r[k] is None for k in ('mask','mask_area','bbox_xyxy','visible','confidence','seed_frame'))
                assert r['timestamp']==before['timestamp'] and r['visibility']=='unknown'
                assert r['provenance']['review_sha256']==meta['review_sha256'] and r['provenance']['event_frame']<=i
                assert r['robot_subtraction'] is False and r['human_confirmed'] is False
            else:assert r==before
    assert len(seen)==5890
    manifest=json.loads((a.root/'final_manifest.json').read_text());assert len(manifest['streams'])==6
    for item in manifest['streams']:assert sha(item['path'])==item['sha256']
    for c in cfg['spec']['cases']:
        source=folder/'objects.jsonl.gz' if c['case_id']=='right_banana_occlusion' else a.root/c['case_id']/'repair/objects.jsonl.gz'
        for _,rr in rows_at(source).items():
            r=next(r for r in rr if r['object_id']==c['object_id'])
            counts['unknown' if r['mask'] is None else ('visible' if r['visible'] else 'empty')]+=1
    write_json(a.root/'final_validation.json',dict(status='PASS',semantic_acceptance=False,
        final_streams=6,camera_frames=sum(s['frames'] for s in manifest['streams']),
        boundary_changes=12,boundary_passthrough_rows=5878,target_rows=counts,original_source_unchanged=True))
    print((a.root/'final_validation.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['contacts','package','finalcheck']);p.add_argument('--root',type=Path,default=DEFAULT)
    a=p.parse_args();{'contacts':contacts,'package':package,'finalcheck':finalcheck}[a.phase](a)
