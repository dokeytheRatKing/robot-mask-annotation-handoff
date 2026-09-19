"""Materialize assistant decisions and portable QA for the bounded reentry study."""
import argparse
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from annotate import BASE, PROJECT, sha, write_json
from full_episode_reentry import DEFAULT, rows_at
from seed_bank_transfer import original_frame
from seed_bank_transfer_qa import labelled


def prompts(a):
    source=BASE/'config/full_reentry_assistant_20260918.json'
    decisions=json.loads(source.read_text());rows=[]
    for d in decisions:
        common=dict(case_id=f'episode_{d["episode"]:06d}_{d["camera"]}',frame_idx=d['frame'],
                    note=d.get('note','Assistant inspected original RGB, no GT mask reference.'))
        for oid in d.get('clear',[]):rows.append(dict(common,object_id=oid,status='not_visible'))
        for oid in d.get('uncertain',[]):rows.append(dict(common,object_id=oid,status='uncertain'))
        for oid,spec in d.get('visible',{}).items():rows.append(dict(common,object_id=int(oid),status='visible',**spec))
    out=a.root/('assistant_prompts.json' if a.revision=='seeds_r1' else f'assistant_prompts_{a.revision}.json');assert not out.exists()
    write_json(out,rows);print(dict(prompts=len(rows),frames=len(decisions),sha256=sha(out)))


def package(a):
    target=PROJECT/'deliverables/astribot_full_episode_reentry_20260918.zip'
    assert not target.exists()
    files=[p for p in a.root.rglob('*') if p.is_file()
           and 'frames' not in p.relative_to(a.root).parts
           and not any(part.startswith('qa_interrupted_') for part in p.relative_to(a.root).parts)]
    files += [BASE/n for n in ('full_episode_reentry.py','reentry_artifacts.py','validate_full_reentry.py','test_full_episode_reentry.py')]
    files += [BASE/'config/full_reentry_assistant_20260918.json',PROJECT/'docs/full_episode_reentry_report_20260918.md']
    files += list((PROJECT/'runs').glob('reentry_*_20260918.log'))
    manifest=[dict(path=str(p.relative_to(PROJECT)),size=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(PROJECT)))
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    write_json(a.root/'delivery.json',dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(files)+1,crc='PASS'))
    print(json.dumps(json.loads((a.root/'delivery.json').read_text()),indent=2))


def preview(a):
    folder=a.root/a.revision;rows=json.loads((folder/'seeds.json').read_text());tiles=[]
    for r in rows:
        if not r['visible']:continue
        path=folder/f'{r["case_id"]}_{r["frame_idx"]}_{r["object_id"]}.jpg'
        image=cv2.imread(str(path));assert image is not None
        tiles.append(image)
    for start in range(0,len(tiles),9):
        page=tiles[start:start+9]
        while len(page)%3:page.append(np.zeros_like(page[0]))
        out=folder/f'contact_{start//9}.jpg';assert not out.exists()
        cv2.imwrite(str(out),np.concatenate([np.concatenate(page[i:i+3],1) for i in range(0,len(page),3)],0))
    print('positive_masks',len(tiles))


def snapshots(a):
    cfg=json.loads((a.root/'config.json').read_text());dest=a.root/'spotchecks';dest.mkdir(exist_ok=False)
    wanted={'episode_001662_left_wrist':[149], 'episode_001662_right_wrist':[149],
        'episode_002503_left_wrist':[270,390,630,870,990,1158],
        'episode_002503_right_wrist':[638,668,748,898,1008,1158]}
    for c in cfg['cases']:
        if c['case_id'] not in wanted:continue
        layers={m:rows_at(a.root/c['case_id']/m/'objects.jsonl.gz') for m in ('baseline','recovery')}
        for idx in wanted[c['case_id']]:
            image=original_frame(c,idx);cells=[]
            for rows,title in [([],f'{c["camera"]} RGB {idx}'),(layers['baseline'][idx],'Baseline'),(layers['recovery'][idx],'Assistant recovery')]:
                rgb,legend=labelled(image,rows,title)
                legend=np.pad(legend,((0,len(c['object_ids'])*16-legend.shape[0]),(0,0),(0,0)))
                cells.append(np.concatenate([rgb,legend],0))
            cv2.imwrite(str(dest/f'{c["case_id"]}_{idx}.jpg'),np.concatenate(cells,1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prompts','package','preview','snapshots']);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--revision',default='seeds_r1')
    a=p.parse_args();{'prompts':prompts,'package':package,'preview':preview,'snapshots':snapshots}[a.phase](a)
