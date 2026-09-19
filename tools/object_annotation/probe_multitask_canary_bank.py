"""Read-only, episode-excluded identity retrieval for the fixed canary score frames."""
import json
import cv2
import torch
from annotate import PROJECT, sha, write_json
from full_episode_reentry import rows_at
from identity_guard import Appearance
from masks import decode
from seed_bank import SeedBank, crops
from multitask_reseed_canary import ROOT, load, read_image, MODES


def main():
    cfg=load(ROOT);bankpath=PROJECT/'annotations/identity_seed_bank_20260918_curated'
    bank=SeedBank(bankpath);encoder=Appearance();results=[]
    for c in cfg['cases']:
        idx=c['eval_frame'];image=read_image(ROOT,c,idx)
        paths={'frozen_production':c['original_objects']['path'],
            **{m:ROOT/c['case_id']/'predictions'/f'{m}.jsonl.gz' for m in MODES}}
        for mode,path in paths.items():
            for r in rows_at(path)[idx]:
                retrieval=None;passed=None
                if r['mask'] is not None and decode(r['mask']).any():
                    rgb,masked,_,_=crops(image,decode(r['mask']));emb=encoder.embed([rgb,masked]).cpu().numpy()
                    retrieval=bank.query(emb[0],emb[1],r['object_id'],c['camera'],exclude_episode=c['episode']['episode_id'],mode='masked')
                    sim,margin=retrieval['similarity'],retrieval['margin']
                    passed=sim is not None and margin is not None and sim>=.50 and margin>=.05
                    for eid in retrieval['nearest_exemplar_ids']:
                        assert next(e for e in bank.entries if e['exemplar_id']==eid)['episode_id']!=c['episode']['episode_id']
                results.append(dict(case_id=c['case_id'],frame_idx=idx,object_id=r['object_id'],mode=mode,
                    retrieval=retrieval,existing_gate_passes=passed,presence_confidence=r['confidence']))
    out=ROOT/'bank_probe.json';assert not out.exists()
    write_json(out,dict(bank_sha256=sha(bankpath/'bank.json'),embedding_sha256=sha(bankpath/'embeddings.npz'),
        encoder_sha256=sha(PROJECT/'models/dinov2/dinov2_vits14_pretrain.pth'),code_sha256=sha(__file__),rows=results,
        policy='Diagnostic only. Existing cosine .50 / margin .05, same-camera preferred, query episode excluded. No bank expansion, threshold tuning or mask mutation.'))
    print('BANK PROBE',len(results),'records',sum(r['retrieval'] is not None for r in results),'queries',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);cv2.setNumThreads(1);main()
