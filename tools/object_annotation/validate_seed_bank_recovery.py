"""Independently verify shared candidate decisions, causal seeds and bank exclusions."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from annotate import BASE, sha, write_json
from masks import decode, bbox
from seed_bank import SeedBank
from seed_bank_recovery import MODES, DEFAULT
from seed_bank_transfer import choose


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    a=p.parse_args();cfg=json.loads((a.root/'run_config.json').read_text())
    assert sha(Path(cfg['parent'])/'run_config.json')==cfg['parent_config_sha256']
    parent=json.loads((Path(cfg['parent'])/'run_config.json').read_text())
    assert sha(Path(cfg['bank'])/'bank.json')==cfg['bank_sha256']
    for name,digest in cfg['code'].items():assert sha(BASE/name)==digest
    bank=SeedBank(Path(cfg['bank']));entries={e['exemplar_id']:e for e in bank.entries}
    counts=Counter();detection_frames={};unknown=Counter()
    for case in cfg['cases']:
        if case['kind']=='unseen_episode':
            assert case['episode']['episode_id'] not in parent['excluded_source_episodes']
        directory=a.root/case['case_id'];meta=json.loads((directory/'complete.json').read_text())
        assert meta['case']==case and meta['status']=='COMPLETE'
        assert sha(meta['robot_source']['path'])==meta['robot_source']['sha256']
        for name in ('events','checks','candidates'):
            path=directory/(name+('.jsonl' if name=='candidates' else '.json'))
            assert sha(path)==meta[name+'_sha256']
        checks=json.loads((directory/'checks.json').read_text());anchors={c['frame_idx'] for c in checks}
        assert len(anchors)==len(checks) and case['start'] in anchors and case['eval_frame'] not in anchors
        assert all(case['start']<=idx<=case['end'] and (idx-case['start'])%5==0 for idx in anchors)
        pool=[json.loads(line) for line in (directory/'candidates.jsonl').read_text().splitlines()]
        assert len({r['candidate_id'] for r in pool})==len(pool)
        byid={r['candidate_id']:r for r in pool}
        for r in pool:
            assert r['frame_idx'] in anchors and r['object_id'] in case['object_ids']
            binary=decode(r['mask']);assert bbox(binary)==r['bbox_xyxy'] and int(binary.sum())==r['mask_area']
            retrieval=r['retrieval']
            refs=retrieval['nearest_exemplar_ids']+([retrieval['competitor_exemplar_id']] if retrieval.get('competitor_exemplar_id') else [])
            for eid in refs:
                e=entries[eid];assert e['bank_status']=='active' and e['episode_id']!=case['episode']['episode_id']
                counts['episode_excluded_references']+=1
        events=json.loads((directory/'events.json').read_text());index={}
        for e in events:
            key=(e['mode'],e['frame_idx'],e['object_id']);assert key not in index;index[key]=e
            assert e['frame_idx'] in anchors
            if e['mode'].startswith('single_'):assert e['frame_idx']==case['start']
            candidates=[r for r in pool if r['frame_idx']==e['frame_idx'] and r['object_id']==e['object_id']]
            chosen,reason=choose(candidates,'bank_rank' if e['mode'].endswith('bank') else 'detector',True)
            assert e['candidate_id']==(chosen['candidate_id'] if chosen else None)
            assert e['selection_reason']==reason
            if e['action']=='seed':assert chosen is not None
            if e['action']=='terminate':assert chosen is None and e['misses']>=3 and e['reason']!='robot_overlap_warning'
            counts[e['mode']+'/'+e['action']+'/'+e['reason']]+=1
        expected={(mode,idx,oid) for mode in MODES for idx in (anchors if mode.startswith('recovery_') else [case['start']]) for oid in case['object_ids']}
        assert set(index)==expected
        for mode in MODES:
            path=directory/f'{mode}.jsonl.gz';assert sha(path)==meta['output_sha256'][mode]
            current={oid:None for oid in case['object_ids']};seen=set();times={}
            with gzip.open(path,'rt') as f:
                for r in map(json.loads,f):
                    idx=r['frame_idx'];oid=r['object_id'];key=(idx,oid)
                    assert key not in seen;seen.add(key)
                    event=index.get((mode,idx,oid))
                    if event and event['action'] in ('seed','terminate'):
                        current[oid]=event if event['action']=='seed' else None
                    assert r['mode']==mode and r['human_confirmed'] is False
                    assert np.isfinite(r['timestamp']);times[idx]=r['timestamp']
                    if r['mask'] is None:
                        assert current[oid] is None and r['visible'] is None and r['visibility']=='unknown'
                        unknown[mode]+=1
                    else:
                        assert current[oid] is not None and r['seed_frame']==current[oid]['frame_idx']<=idx
                        provenance=r['provenance'];assert provenance['candidate_id']==current[oid]['candidate_id']
                        assert provenance['bank']==byid[provenance['candidate_id']]['retrieval']
                        assert provenance['bank_sha256']==cfg['bank_sha256'] and provenance['robot_subtraction'] is False
                        mask=decode(r['mask']);assert mask.shape==(r['image_height'],r['image_width'])
                        assert bbox(mask)==r['bbox_xyxy'] and int(mask.sum())==r['mask_area'] and bool(mask.any())==r['visible']
                        assert np.isfinite(r['confidence']) and 0<=r['confidence']<=1
                    counts['rows']+=1
            assert seen=={(idx,oid) for idx in range(case['start'],case['end']+1) for oid in case['object_ids']}
            assert np.all(np.diff([times[idx] for idx in sorted(times)])>=0)
        detection_frames[case['case_id']]=sorted(anchors);counts['cases']+=1
    result=dict(status='PASS',counts=dict(counts),unknown_rows=dict(unknown),shared_detection_frames=detection_frames,
                config_sha256=sha(a.root/'run_config.json'),scope='Causal provenance, geometry, common pool, query exclusion; NOT semantic acceptance')
    write_json(a.root/'independent_validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
