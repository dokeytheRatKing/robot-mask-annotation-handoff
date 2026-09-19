"""Validate bank assets, normalized features, provenance, and pilot exclusion."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from annotate import sha, write_json
from masks import bbox, decode
from seed_bank import SeedBank


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--pilot',type=Path)
    a=p.parse_args();bank=SeedBank(a.bank);checked=set()
    for i,e in enumerate(bank.entries):
        assert e['embedding_index']==i and e['human_confirmed']
        mask=decode(e['mask']);assert bbox(mask)==e['bbox_xyxy'] and int(mask.sum())==e['mask_area']
        assert mask.shape==(e['image_height'],e['image_width'])
        for filename,relative in e['assets'].items():
            assert sha(a.bank/relative)==e['asset_sha256'][filename]
        saved=cv2.imread(str(a.bank/e['assets']['mask.png']),cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(saved))<={0,255} and np.array_equal(saved>0,mask)
        x0,y0,x1,y1=e['crop_bbox_xyxy']
        cropmask=cv2.imread(str(a.bank/e['assets']['crop_mask.png']),0)>0
        assert np.array_equal(cropmask,mask[y0:y1,x0:x1])
        rgb=cv2.imread(str(a.bank/e['assets']['rgb.png']))
        cut=cv2.imread(str(a.bank/e['assets']['masked_rgb.png']))
        assert np.array_equal(rgb[cropmask],cut[cropmask]) and (cut[~cropmask]==127).all()
        for path,digest in [(e['rgb_path'],e['provenance']['source_image_sha256']),
                            (e['source_manifest'],e['source_manifest_sha256'])]:
            if path not in checked:assert sha(path)==digest;checked.add(path)
    result=dict(status='PASS',exemplars=len(bank.entries),active=sum(e['bank_status']=='active' for e in bank.entries),
                source_files_verified=len(checked),unit_embeddings=True,pixel_accuracy='not asserted')
    if a.pilot:
        cfg=json.loads((a.pilot/'run_config.json').read_text());assert cfg['bank_sha256']==sha(a.bank/'bank.json')
        meta=json.loads((a.pilot/'metrics.json').read_text())
        assert meta['candidates_sha256']==sha(a.pilot/'candidates.jsonl')
        assert meta['queries_sha256']==sha(a.pilot/'queries.jsonl')
        lookup={e['exemplar_id']:e for e in bank.entries};count=0
        candidates={}
        for c in map(json.loads,(a.pilot/'candidates.jsonl').read_text().splitlines()):
            assert c['candidate_id'] not in candidates;candidates[c['candidate_id']]=c
            assert bbox(decode(c['mask']))==c['bbox_xyxy']
            for r in c['retrieval'].values():
                ids=r['nearest_exemplar_ids']+([r['competitor_exemplar_id']] if r.get('competitor_exemplar_id') else [])
                for eid in ids:
                    e=lookup[eid];assert e['episode_id']!=c['episode_id'] and e['bank_status']=='active';count+=1
        seen=set()
        for q in map(json.loads,(a.pilot/'queries.jsonl').read_text().splitlines()):
            assert q['query_id'] not in seen;seen.add(q['query_id'])
            sets=[]
            for mode,m in q['methods'].items():
                sets.append(set(m['ranking']))
                assert len(m['ranking'])==len(sets[-1])
                for cid in m['ranking']:assert candidates[cid]['query_id']==q['query_id']
                assert m['candidate_id']==(m['ranking'][0] if m['ranking'] else None)
                scores=[candidates[cid]['scores'][mode] for cid in m['ranking']]
                assert scores==sorted(scores,reverse=True)
            assert all(s==sets[0] for s in sets)
        result.update(pilot_targets=len(seen),candidates=len(candidates),episode_exclusion_checks=count,
                      identical_candidate_pools=True)
    target=(a.pilot if a.pilot else a.bank)/'validation.json'
    write_json(target,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
