"""Render exact completed recovery predictions for assistant inspection."""
import argparse
from collections import defaultdict
import gzip
import json

import cv2
import numpy as np

from annotate import sha
from seed_bank_recovery import DEFAULT, MODES
from seed_bank_transfer import original_frame, old_rows
from seed_bank_transfer_qa import labelled


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True)
    p.add_argument('--frames',type=int,nargs='+');a=p.parse_args()
    cfg=json.loads((DEFAULT/'run_config.json').read_text())
    case=next(c for c in cfg['cases'] if c['case_id']==a.case)
    src=DEFAULT/a.case;meta=json.loads((src/'complete.json').read_text());layers={}
    for mode in MODES:
        path=src/f'{mode}.jsonl.gz';assert sha(path)==meta['output_sha256'][mode]
        layers[mode]=defaultdict(list)
        with gzip.open(path,'rt') as f:
            for r in map(json.loads,f):layers[mode][r['frame_idx']].append(r)
    dest=DEFAULT/'inspection'/a.case;dest.mkdir(parents=True,exist_ok=True)
    for idx in a.frames or [case['start'],(case['start']+case['end'])//2,case['end']]:
        assert case['start']<=idx<=case['end']
        image=original_frame(case,idx);robot,_=old_rows(case['episode'],case['camera'],idx,idx,'robot');cells=[]
        for rows,title in [([],f'{a.case} {idx} RGB'),(robot[idx],'Robot QA only'),*[(layers[m][idx],m) for m in MODES]]:
            rgb,legend=labelled(image,rows,title)
            legend=np.pad(legend,((0,max(1,len(case['object_ids']))*16-legend.shape[0]),(0,0),(0,0)))
            cells.append(np.concatenate([rgb,legend],0))
        canvas=np.concatenate([np.concatenate(cells[:3],1),np.concatenate(cells[3:],1)],0)
        target=dest/f'{idx}.jpg';assert not target.exists();assert cv2.imwrite(str(target),canvas)
        print(target)


if __name__=='__main__':main()
