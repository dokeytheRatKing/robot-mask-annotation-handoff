"""Rerank the SAME frozen detector/SAM candidates with a curated bank."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from annotate import sha, write_json
from identity_guard import Appearance
from masks import decode
from seed_bank import SeedBank, crops, frame_key, rank_score, read_image
from seed_bank_pilot import selected_queries, summarize, tint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True,type=Path)
    p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);(a.output/'qa').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    bank=SeedBank(a.bank);encoder=Appearance();references=selected_queries();checked=set()
    queries=[json.loads(s) for s in (a.parent/'queries.jsonl').read_text().splitlines()]
    byframe=defaultdict(list);byquery=defaultdict(list)
    for q in queries:byframe[frame_key(q)].append(q)
    for c in map(json.loads,(a.parent/'candidates.jsonl').read_text().splitlines()):byquery[c['query_id']].append(c)
    config=json.loads((a.parent/'run_config.json').read_text())
    config.update(bank=str(a.bank),bank_sha256=sha(a.bank/'bank.json'),parent=str(a.parent),
        parent_candidates_sha256=sha(a.parent/'candidates.jsonl'),parent_queries_sha256=sha(a.parent/'queries.jsonl'),
        curation_review_sha256=sha(a.bank/'assistant_visual_review.json'),rescore_code_sha256=sha(__file__))
    write_json(a.output/'run_config.json',config)
    started=time.perf_counter();results=[]
    with (a.output/'queries.jsonl').open('x') as qf,(a.output/'candidates.jsonl').open('x') as cf:
        for key,frame_queries in byframe.items():
            labels={r['object_id']:r for r in references[key]}
            image=read_image(next(iter(labels.values())),checked)
            candidates=[c for q in frame_queries for c in byquery[q['query_id']]]
            masks=[decode(c['mask']) for c in candidates]
            pairs=[crops(image,m) for m in masks]
            qr=encoder.embed([x[0] for x in pairs]).cpu().numpy()
            qm=encoder.embed([x[1] for x in pairs]).cpu().numpy()
            for i,c in enumerate(candidates):
                c['retrieval']={mode:bank.query(qr[i],qm[i],c['target_id'],key[1],exclude_episode=key[0],mode=mode)
                                for mode in ('rgb','masked','hybrid')}
                c['scores']={mode:rank_score(c['detector_score'],r) for mode,r in c['retrieval'].items()}
                c['scores']['detector']=c['detector_score']
                cf.write(json.dumps(c,allow_nan=False)+'\n')
            for q in frame_queries:
                oid=q['object_id'];options=byquery[q['query_id']];methods={};chosen={}
                for mode in ('detector','rgb','masked','hybrid'):
                    ranked=sorted(options,key=lambda c:(-c['scores'][mode],c['candidate_id']))
                    first=ranked[0] if ranked else None;chosen[mode]=first
                    accepted=first is not None and first['detector_score']>=.30
                    if mode!='detector' and first:
                        r=first['retrieval'][mode]
                        accepted=accepted and r['similarity'] is not None and r['similarity']>=.50 and r['margin'] is not None and r['margin']>=.05 and oid<1000
                    methods[mode]=dict(candidate_id=first['candidate_id'] if first else None,
                        ranking=[c['candidate_id'] for c in ranked],accepted=bool(accepted),
                        **(first['reference_metrics'] if first else dict(iou=0,precision=0,recall=0)))
                assert methods['detector']==q['methods']['detector']
                q=dict(q,methods=methods,reference_available=any(e['episode_id']!=key[0] and e['bank_status']=='active' and
                    oid in (e.get('merged_object_ids') or [e['object_id']]) for e in bank.entries))
                results.append(q);qf.write(json.dumps(q,allow_nan=False)+'\n')
                if oid in (0,1,4,5,10,20,21) or methods['detector']['candidate_id']!=methods['hybrid']['candidate_id']:
                    gt=decode(labels[oid]['mask']);panels=[tint(image,None,f'{q["query_id"]} RGB {key[1]}',(0,0,0)),tint(image,gt,'Accepted reference',(0,220,0))]
                    for mode,color in [('detector',(0,180,255)),('hybrid',(255,180,0))]:
                        c=chosen[mode];panels.append(tint(image,decode(c['mask']) if c else None,f'{mode} IoU={methods[mode]["iou"]:.3f}',color))
                    cv2.imwrite(str(a.output/'qa'/f'{q["query_id"]}.jpg'),np.concatenate(panels,1))
            print(f'rescored {key}',flush=True)
    groups=defaultdict(list)
    for r in results:groups[f'{r["object_id"]}/{r["camera"]}'].append(r)
    write_json(a.output/'metrics.json',dict(summary=summarize(results),frames=len(byframe),seconds=time.perf_counter()-started,
        by_identity_camera={k:summarize(v)['all'] for k,v in groups.items()},
        candidates_sha256=sha(a.output/'candidates.jsonl'),queries_sha256=sha(a.output/'queries.jsonl')))
    print(json.dumps(summarize(results),indent=2),flush=True)


if __name__=='__main__':main()
