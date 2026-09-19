"""Final bounded seam correction; preserve r1/r2 and all untouched records."""
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
    root=DEFAULT;cfg=load(root);dest=root/'final_r3';dest.mkdir(exist_ok=False)
    source=root/'boundary_r2/episode_002503_right_wrist.jsonl.gz'
    before=rows_at(source);after=copy.deepcopy(before)
    spec=dict(reviewer='active_session_assistant',human_confirmed=False,new_human_requests=0,
        original_frames_viewed=[644,645,700,782,784,786,790,791,800,820],
        overrides=[dict(start=644,end=644,reason='banana_hidden_at_anchor'),
                   dict(start=791,end=819,reason='continue_unknown_until_reviewed_reappearance')],
        note='Frames791/800 visibly occluded;820 visible again. Unreviewed intermediate frames are UNKNOWN, not confirmed absent. Resume820 uses original track, not a new SAM2 seed.')
    write_json(dest/'review.json',spec);expected={i for op in spec['overrides'] for i in range(op['start'],op['end']+1)}
    for op in spec['overrides']:
        for i in range(op['start'],op['end']+1):
            r=next(r for r in after[i] if r['object_id']==2)
            r.update(mask=None,mask_area=None,bbox_xyxy=None,visible=None,confidence=None,seed_frame=None,
                visibility='unknown',mask_status='unknown',mode='semantic_seam_r3',human_confirmed=False,
                provenance=dict(kind='assistant_seam_clear',event_frame=op['start'],parent_sha256=sha(source),
                    review_sha256=sha(dest/'review.json'),causal=True),suspicious_flags=['assistant_repair_not_gt','unknown_visibility_interval'])
    path=dest/'episode_002503_right_wrist.jsonl.gz';changed=0;unchanged=0
    with gzip.open(path,'wt') as f:
        for i,rows in after.items():
            for r in rows:
                old=next(x for x in before[i] if x['object_id']==r['object_id'])
                if i in expected and r['object_id']==2:
                    assert r['mask'] is None and r['visible'] is None and r['visibility']=='unknown'
                    assert r['timestamp']==old['timestamp'];changed+=1
                else:assert r==old;unchanged+=1
                f.write(json.dumps(r,allow_nan=False)+'\n')
    assert changed==30 and unchanged==5860
    c=next(c for c in cfg['spec']['cases'] if c['case_id']=='right_banana_occlusion')
    original=rows_at(PARENT/c['source_case']/'recovery/objects.jsonl.gz')
    writer=Video(dest/'right_banana_comparison.mp4',30)
    snapshots=[638,644,646,668,700,709,710,720,748,780,784,786,790,791,800,819,820,825,830]
    for i in range(630,831):
        cells=[];image=image_at(c,i)
        for rr,title in [([],f'RGB {i}'),(original[i],'Original recovery'),(after[i],'Final bounded repair')]:
            rgb,legend=labelled(image,[r for r in rr if r['object_id']==2],title);cells.append(np.concatenate([rgb,legend],0))
        canvas=np.concatenate(cells,1);writer.add(canvas)
        if i in snapshots:cv2.imwrite(str(dest/f'{i:05d}.jpg'),canvas)
    writer.close();cap=cv2.VideoCapture(str(dest/'right_banana_comparison.mp4'));n=0
    while True:
        ok,image=cap.read()
        if not ok:break
        assert image.shape==writer.shape;n+=1
    cap.release();assert n==201
    manifest=json.loads((root/'final_manifest.json').read_text())
    for r in manifest['streams']:
        if r['case_id']==c['source_case']:r.update(path=str(path),sha256=sha(path))
    manifest['authoritative_revision']='final_r3';write_json(root/'final_manifest_r3.json',manifest)
    # Independently read back all keys and ensure only explicitly allowed target intervals differ from the original full-run parent.
    count=0;mutated=0;pass_through=0
    for item in manifest['streams']:
        assert sha(item['path'])==item['sha256'];new=rows_at(item['path']);old=rows_at(PARENT/item['case_id']/'recovery/objects.jsonl.gz')
        assert new.keys()==old.keys() and len(new)==item['frames']
        for i,rows in new.items():
            assert len(rows)==len(old[i]) and len({r['object_id'] for r in rows})==len(rows)
            for r in rows:
                previous=next(x for x in old[i] if x['object_id']==r['object_id']);count+=1
                permitted=(item['case_id']=='episode_002503_head' and r['object_id']==2 and i<=547) or (
                    item['case_id']=='episode_002503_right_wrist' and ((r['object_id']==2 and 630<=i<=819) or(r['object_id']==20 and 880<=i<=1139)))
                if not permitted:assert r==previous;pass_through+=1
                else:
                    mutated+=1;assert r['timestamp']==previous['timestamp']
                    assert r['human_confirmed'] is False and r['robot_subtraction'] is False
    assert count==18393 and mutated==998
    write_json(dest/'validation.json',dict(status='PASS',semantic_acceptance=False,r2_parent_sha256=sha(source),
        r3_sha256=sha(path),r3_changed_from_r2=changed,r3_unchanged_from_r2=unchanged,
        full_manifest_rows=count,modified_target_rows=mutated,unchanged_rows=pass_through,
        decoded_frames=n,video_sha256=sha(dest/'right_banana_comparison.mp4'),code_sha256=sha(__file__)))
    print((dest/'validation.json').read_text(),flush=True)


if __name__=='__main__':cv2.setNumThreads(1);main()
