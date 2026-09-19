"""Post-hoc candidate ceiling from earlier held-out-frame caches; never used as seeds."""
from collections import defaultdict
import json

import numpy as np

from annotate import PROJECT, sha, write_json
from seed_bank_recovery import DEFAULT


def main():
    root=PROJECT/'annotations/identity_seed_bank_pilot_20260918_curated'
    cfg=json.loads((DEFAULT/'run_config.json').read_text())
    old=json.loads((root/'run_config.json').read_text())
    meta=json.loads((root/'metrics.json').read_text())
    assert old['bank_sha256']==cfg['bank_sha256']
    for name in ('queries','candidates'):assert sha(root/(name+'.jsonl'))==meta[name+'_sha256']
    targets={(c['episode']['episode_id'],c['camera'],c['eval_frame']) for c in cfg['cases'] if c['eval_frame'] is not None}
    queries=[r for r in map(json.loads,(root/'queries.jsonl').read_text().splitlines())
             if (r['episode_id'],r['camera'],r['frame_idx']) in targets and r['object_id']<1000]
    pool=defaultdict(list)
    for r in map(json.loads,(root/'candidates.jsonl').read_text().splitlines()):pool[r['query_id']].append(r)
    rows=[]
    for q in queries:
        candidates=pool[q['query_id']];eligible=[c for c in candidates if c['detector_score']>=.30]
        ranked={m:sorted(eligible,key=lambda c:(-c['scores'][m],c['candidate_id'])) for m in ('detector','masked')}
        rows.append(dict(query_id=q['query_id'],camera=q['camera'],object_id=q['object_id'],gt_visible=q['gt_visible'],
            raw_oracle_iou=max([c['reference_metrics']['iou'] for c in candidates],default=0),
            eligible_oracle_iou=max([c['reference_metrics']['iou'] for c in eligible],default=0),
            eligible_count=len(eligible),
            detector_iou=ranked['detector'][0]['reference_metrics']['iou'] if ranked['detector'] else 0,
            bank_iou=ranked['masked'][0]['reference_metrics']['iou'] if ranked['masked'] else 0))
    pos=[r for r in rows if r['gt_visible']]
    result=dict(source=str(root),source_candidate_sha256=meta['candidates_sha256'],source_queries_sha256=meta['queries_sha256'],
        purpose='Post-hoc diagnostic only; these scoring-frame proposals were NEVER given to the recovery runner.',
        visible=len(pos),absent=len(rows)-len(pos),
        valid_raw_oracle=sum(r['raw_oracle_iou']>=.5 for r in pos),
        valid_eligible_oracle=sum(r['eligible_oracle_iou']>=.5 for r in pos),
        valid_detector=sum(r['detector_iou']>=.5 for r in pos),valid_bank=sum(r['bank_iou']>=.5 for r in pos),
        mean_eligible_oracle_iou=float(np.mean([r['eligible_oracle_iou'] for r in pos])),rows=rows)
    write_json(DEFAULT/'candidate_ceiling.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
