"""Read-only frozen-bank evidence on predeclared non-seed failure checks."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import torch

from annotate import PROJECT, sha, write_json
from full_episode_reentry import rows_at
from identity_guard import Appearance
from masks import decode
from seed_bank import SeedBank, crops
from semantic_interval_repair import load, image_at, PARENT, DEFAULT


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args()
    cfg=load(a.root);dest=a.root/'bank_probe';dest.mkdir(exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    bankpath=PROJECT/'annotations/identity_seed_bank_20260918_curated';bank=SeedBank(bankpath);encoder=Appearance();result=[]
    for c in cfg['spec']['cases']:
        source=rows_at(PARENT/c['source_case']/'recovery/objects.jsonl.gz');new=rows_at(a.root/c['case_id']/'repair/objects.jsonl.gz')
        for i in c['checks']:
            image=image_at(c,i)
            for mode,rows in [('previous_recovery',source),('local_repair',new)]:
                r=next(r for r in rows[i] if r['object_id']==c['object_id']);retrieval=None;passes=None
                if r['mask'] is not None and decode(r['mask']).any():
                    rgb,cutout,_,_=crops(image,decode(r['mask']));emb=encoder.embed([rgb,cutout]).cpu().numpy()
                    retrieval=bank.query(emb[0],emb[1],c['object_id'],r['camera'],exclude_episode=r['episode_id'],mode='masked')
                    s,m=retrieval['similarity'],retrieval['margin'];passes=s is not None and m is not None and s>=.50 and m>=.05
                    for eid in retrieval['nearest_exemplar_ids']:
                        assert next(e for e in bank.entries if e['exemplar_id']==eid)['episode_id']!=r['episode_id']
                result.append(dict(case_id=c['case_id'],frame_idx=i,object_id=c['object_id'],mode=mode,
                    mask_area=r['mask_area'],presence_confidence=r['confidence'],retrieval=retrieval,
                    existing_similarity_margin_gate=passes,decision='diagnostic_only_no_mask_change'))
    out=dict(bank_sha256=sha(bankpath/'bank.json'),encoder_sha256=sha(PROJECT/'models/dinov2/dinov2_vits14_pretrain.pth'),
        code_sha256=sha(__file__),rows=result,policy='Frozen thresholds similarity .50 / margin .05. No calibration, detector gating, mutation, or corpus accuracy claim.')
    write_json(dest/'results.json',out)
    with (dest/'results.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['case_id','frame_idx','mode','mask_area','presence_confidence','similarity','margin','gate']);writer.writeheader()
        for r in result:
            writer.writerow({k:r[k] for k in ['case_id','frame_idx','mode','mask_area','presence_confidence']} |
                dict(similarity=(r['retrieval'] or {}).get('similarity'),margin=(r['retrieval'] or {}).get('margin'),gate=r['existing_similarity_margin_gate']))
    print('PROBE COMPLETE',len(result),flush=True)


if __name__=='__main__':main()
