"""Dense contracts and frozen-reference exclusion for bounded transfer pilot."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from annotate import sha,write_json
from masks import bbox,decode
from seed_bank import SeedBank
from seed_bank_transfer import MODES,choose


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.root/'run_config.json').read_text());bank=SeedBank(Path(cfg['bank']))
    qa=json.loads((a.root/'qa_manifest.json').read_text())
    assert qa['status']=='PASS'
    videos={v['case_id']:v for v in qa['videos']}
    assert sha(Path(cfg['bank'])/'bank.json')==cfg['bank_sha256']
    assert sha(a.root/'bank_acceptance.json')==cfg['acceptance_sha256']
    lookup={e['exemplar_id']:e for e in bank.entries};counts=Counter();expected_rows=0
    for case in cfg['cases']:
        directory=a.root/case['case_id'];meta=json.loads((directory/'complete.json').read_text())
        assert meta['status']=='COMPLETE' and meta['case']==case
        if case['kind']=='unseen_episode':assert case['episode']['episode_id'] not in cfg['excluded_source_episodes']
        if case['eval_frame'] is not None:assert case['eval_frame'] not in case['seed_frames']
        for key in ('old_source','robot_source'):assert sha(meta[key]['path'])==meta[key]['sha256']
        assert sha(directory/'candidates.jsonl')==meta['candidates_sha256']
        assert sha(directory/'selections.json')==meta['selections_sha256']
        candidates=[json.loads(s) for s in (directory/'candidates.jsonl').read_text().splitlines()]
        selections=json.loads((directory/'selections.json').read_text());bykey={}
        assert len({c['candidate_id'] for c in candidates})==len(candidates)
        for c in candidates:
            assert c['frame_idx'] in case['seed_frames'] and c['object_id'] in case['object_ids']
            mask=decode(c['mask']);assert bbox(mask)==c['bbox_xyxy'] and int(mask.sum())==c['mask_area']
            r=c['retrieval'];refs=r['nearest_exemplar_ids']+([r['competitor_exemplar_id']] if r.get('competitor_exemplar_id') else [])
            for eid in refs:
                entry=lookup[eid]
                assert entry['bank_status']=='active' and entry['episode_id']!=case['episode']['episode_id']
                counts['reference_exclusions']+=1
        for row in selections:
            key=(row['mode'],row['frame_idx'],row['object_id']);assert key not in bykey;bykey[key]=row
            options=[c for c in candidates if c['frame_idx']==row['frame_idx'] and c['object_id']==row['object_id']]
            chosen,reason=choose(options,row['mode'],cfg.get('eligible_first',False))
            assert row['candidate_id']==(chosen['candidate_id'] if chosen else None) and row['reason']==reason
            counts[f'{row["mode"]}/selections/{reason}']+=1
        assert len(bykey)==len(MODES)*len(case['seed_frames'])*len(case['object_ids'])
        for mode in MODES:
            path=directory/f'{mode}.jsonl.gz';assert sha(path)==meta['output_sha256'][mode]
            seen=set();timestamps={}
            with gzip.open(path,'rt') as f:
                for row in map(json.loads,f):
                    key=row['frame_idx'],row['object_id'];assert key not in seen;seen.add(key)
                    assert not row['human_confirmed'] and row['mode']==mode
                    assert case['start']<=row['frame_idx']<=case['end']
                    assert row['object_id'] in case['object_ids'] and row['seed_frame'] in case['seed_frames']
                    assert row['seed_frame']==max(x for x in case['seed_frames'] if x<=row['frame_idx'])
                    ts=row['timestamp'];assert np.isfinite(ts)
                    if row['frame_idx'] in timestamps:assert timestamps[row['frame_idx']]==ts
                    timestamps[row['frame_idx']]=ts
                    seed=bykey[mode,row['seed_frame'],row['object_id']]
                    if row['mask'] is None:
                        assert seed['candidate_id'] is None and row['visible'] is None and row['visibility']=='unknown'
                        counts['unknown_rows']+=1
                    else:
                        assert row['provenance']['candidate_id']==seed['candidate_id']
                        m=decode(row['mask']);assert m.shape==(row['image_height'],row['image_width'])
                        assert bbox(m)==row['bbox_xyxy'] and int(m.sum())==row['mask_area']
                        assert bool(m.any())==row['visible'] and 0<=row['confidence']<=1
                    counts['rows']+=1
            expected=(case['end']-case['start']+1)*len(case['object_ids']);expected_rows+=expected
            assert len(seen)==expected
            times=[timestamps[i] for i in sorted(timestamps)];assert np.all(np.diff(times)>=0)
        video=videos[case['case_id']];assert sha(a.root/video['path'])==video['sha256']
        cap=cv2.VideoCapture(str(a.root/video['path']));decoded=0
        while True:
            ok,image=cap.read()
            if not ok:break
            assert list(image.shape)==video['shape']
            decoded+=1
        cap.release();assert decoded==case['end']-case['start']+1
        counts['qa_frames']+=decoded;counts['cases']+=1
    assert counts['rows']==expected_rows
    result=dict(status='PASS',counts=dict(counts),scope='Structure, same-pool decisions, no source mutation, no episode leakage; NOT semantic acceptance')
    write_json(a.root/'validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
