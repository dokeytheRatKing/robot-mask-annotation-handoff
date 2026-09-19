"""Preserve r1 and apply hash-bound, narrowly scoped assistant UNKNOWN edits."""
import copy
import gzip
import json

import cv2
import numpy as np

from annotate import BASE,sha,write_json
from full_episode_reentry import rows_at
from semantic_interval_repair import DEFAULT,PARENT,load,image_at
from seed_bank_transfer_qa import Video,labelled


def main():
    root=DEFAULT;cfg=load(root);reviewpath=BASE/'config/semantic_boundary_review_20260918.json'
    review=json.loads(reviewpath.read_text());c=next(c for c in cfg['spec']['cases'] if c['case_id']==review['case_id'])
    source=root/c['case_id']/'repair/objects.jsonl.gz';old=rows_at(source);new=copy.deepcopy(old)
    dest=root/'boundary_r2';dest.mkdir(exist_ok=False);expected=set();modified=[]
    for op in review['overrides']:
        assert c['start']<=op['start']<=op['end']<=c['end']
        for i in range(op['start'],op['end']+1):
            assert i not in expected;expected.add(i)
            row=next(r for r in new[i] if r['object_id']==review['object_id'])
            modified.append(dict(frame_idx=i,old_area=row['mask_area'],old_confidence=row['confidence']))
            row.update(mask=None,mask_area=None,bbox_xyxy=None,visible=None,confidence=None,seed_frame=None,
                visibility='unknown',mask_status='unknown',human_confirmed=False,mode='semantic_boundary_r2',
                provenance=dict(kind='assistant_boundary_clear',event_frame=op['start'],decision=op['decision'],
                    review_sha256=sha(reviewpath),parent_sha256=sha(source),causal=True),
                suspicious_flags=['assistant_repair_not_gt','uncertain_visibility' if op['decision'].startswith('uncertain') else 'cleared_after_assistant_review'])
    def save_checked(path,before,after):
        changed=0;unchanged=0
        with gzip.open(path,'wt') as f:
            assert before.keys()==after.keys()
            for i,rows in after.items():
                for row in rows:
                    prev=next(r for r in before[i] if r['object_id']==row['object_id'])
                    if i in expected and row['object_id']==review['object_id']:
                        assert row['mask'] is None and row['visible'] is None and row['visibility']=='unknown'
                        assert row['provenance']['event_frame']<=i and row['timestamp']==prev['timestamp'];changed+=1
                    else:assert row==prev;unchanged+=1
                    f.write(json.dumps(row,allow_nan=False)+'\n')
        assert changed==len(expected)
        return dict(path=str(path.relative_to(root)),sha256=sha(path),changed=changed,unchanged=unchanged)
    clips=save_checked(dest/'objects.jsonl.gz',old,new)
    mergedsource=root/'merged'/f'{c["source_case"]}.jsonl.gz';merged=rows_at(mergedsource);final=copy.deepcopy(merged)
    for i in expected:final[i]=new[i]
    full=save_checked(dest/f'{c["source_case"]}.jsonl.gz',merged,final)
    writer=Video(dest/'comparison.mp4',30);snapshot_frames=[638,644,645,646,649,650,668,700,709,710,720,748,780,783,784,786,790]
    for i,rows in new.items():
        cells=[];image=image_at(c,i)
        for rr,title in [([],f'RGB {i}'),(old[i],'r1 local repair'),(rows,'r2 boundary repair')]:
            rgb,legend=labelled(image,[r for r in rr if r['object_id']==review['object_id']],title)
            cells.append(np.concatenate([rgb,legend],0))
        canvas=np.concatenate(cells,1);writer.add(canvas)
        if i in snapshot_frames:cv2.imwrite(str(dest/f'{i:05d}.jpg'),canvas)
    writer.close();cap=cv2.VideoCapture(str(dest/'comparison.mp4'));n=0
    while True:
        ok,image=cap.read()
        if not ok:break
        assert image.shape==writer.shape;n+=1
    cap.release();assert n==161
    sourcecfg=json.loads((PARENT/'config.json').read_text());manifest=[]
    for sc in sourcecfg['cases']:
        sid=sc['case_id'];path=root/'merged'/f'{sid}.jsonl.gz'
        if sid==c['source_case']:path=dest/f'{sid}.jsonl.gz'
        elif not path.exists():path=PARENT/sid/'recovery/objects.jsonl.gz'
        manifest.append(dict(case_id=sid,path=str(path),sha256=sha(path),frames=sc['episode']['frames'],semantic_acceptance=False))
    write_json(root/'final_manifest.json',dict(status='bounded_candidate_not_gt',streams=manifest,
        note='Two task24 streams locally repaired; other four streams reference frozen parent recovery. No corpus promotion.'))
    write_json(dest/'validation.json',dict(status='PASS',semantic_acceptance=False,clip=clips,full=full,
        reviewed_anchors=[645,784],new_human_requests=0,review=review,review_sha256=sha(reviewpath),
        parent_sha256=sha(source),merged_parent_sha256=sha(mergedsource),modified_rows=modified,
        decoded_frames=n,video_sha256=sha(dest/'comparison.mp4'),code_sha256=sha(__file__)))
    print('BOUNDARY PASS',len(expected),'rows;',n,'video frames',flush=True)


if __name__=='__main__':cv2.setNumThreads(1);main()
