"""Independent validation of the 64-clip source/mask/ROI handoff."""
from collections import Counter
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from annotate import sha, write_json
from calibration64 import ROOT
from masks import decode, bbox


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f]


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text());clips=manifest['clips'];splits=json.loads((ROOT/'splits.json').read_text())
    review=json.loads((ROOT/'review.json').read_text())['clips']
    assert len(clips)==len({c['episode']['episode_id'] for c in clips})==64
    fit={c['episode']['episode_id'] for c in clips if c['split']=='C_fit'}
    diag={c['episode']['episode_id'] for c in clips if c['split']=='C_diag'}
    assert len(fit)==48 and len(diag)==16 and not fit&diag
    assert {c['task_id'] for c in clips if c['split']=='C_fit'}==set(range(1,30))
    assert len(splits['H'])==16 and set(splits['H'])<=set(splits['C_fit'])
    report=[];totals=Counter()
    for c in clips:
        if 'object_ids_override' in review[c['clip_id']]:c=dict(c,object_ids=review[c['clip_id']]['object_ids_override'])
        out=ROOT/'clips'/c['clip_id'];release=out/'release';h,w=c['shape'][:2];n=c['frame_count']
        assert c['episode']['episode_index']<5223 and c['end']<c['episode']['frames'] and c['start']>=0
        assert c['source_horizon_frames']==len(c['prediction_frames']) and len(c['context_frames'])==17
        assert not(set(c['context_frames'])&set(c['prediction_frames']))
        assert c['reference_timestamp']==c['timestamps'][16]
        assert c['physical_horizon_seconds']==c['timestamps'][-1]-c['reference_timestamp']
        for filename,key in [('state_action.npz','state_action_sha256'),('state_action.parquet','parquet_sha256'),('accepted_seeds.json','seed_sha256')]:
            assert sha(out/filename)==c[key]
        data=np.load(out/'state_action.npz',allow_pickle=False)
        assert data['action_absolute_20d'].shape==(n,20) and data['state_raw_25d'].shape==(n,25)
        assert np.isfinite(data['action_absolute_20d']).all() and np.isfinite(data['state_raw_25d']).all()
        assert data['action_valid_dimensions'].shape==(20,) and data['action_valid_dimensions'].all()
        original=pq.read_table(c['episode']['parquet']).slice(c['start'],n)
        assert np.array_equal(np.asarray(original['action'].to_pylist(),np.float32),data['action_absolute_20d'])
        assert np.array_equal(np.asarray(original['observation.state'].to_pylist(),np.float32),data['state_raw_25d'])
        assert np.array_equal(original['frame_index'].to_numpy(),data['frame_idx'])
        for name in data.files:
            if name.startswith('timestamp.'):
                assert np.array_equal(original[name].to_numpy()+c['original_timestamp_start'],data[name])
        assert np.array_equal(data['timestamp.images.'+{'head':'head','left_wrist':'left','right_wrist':'right'}[c['camera']]],c['timestamps'])
        for i,digest in enumerate(c['rgb_sha256']):assert sha(out/'rgb'/f'{i:05d}.png')==digest
        seeds=json.loads((out/'accepted_seeds.json').read_text());sm={(r['frame_idx'],r['object_id']):r for r in seeds}
        seen=set();positive=0;unknown=0;eligible_by_frame={}
        for layer,ids in [('objects',c['object_ids']),('robot_parts',c['robot_part_ids'])]:
            rr=rows(release/f'{layer}.jsonl.gz');assert len(rr)==n*len(ids)
            for r in rr:
                key=r['frame_idx'],r['object_id'];assert key not in seen;seen.add(key)
                assert r['object_id'] in ids and c['start']<=r['frame_idx']<=c['end']
                assert r['timestamp']==c['timestamps'][r['frame_idx']-c['start']]
                assert r['episode_id']==c['episode']['episode_id'] and r['task_id']==c['task_id'] and r['camera']==c['camera']
                if r['mask'] is None:
                    assert r['visible'] is None and not r['training_eligible'];unknown+=1
                    if key in sm and sm[key]['mask'] is not None:
                        assert sm[key]['visible'] is False and 'accepted_negative_discrepancy_unknown' in r['suspicious_flags']
                        assert not r['human_confirmed']
                else:
                    m=decode(r['mask']);assert m.shape==(h,w) and int(m.sum())==r['mask_area'] and bbox(m)==r['bbox_xyxy']
                    assert bool(m.any())==bool(r['visible']);positive+=bool(m.any())
                    if key in sm and r['review_status']!='unreliable_excluded':assert np.array_equal(m,decode(sm[key]['mask']))
                    if r['training_eligible']:
                        assert r['visible'] is True
                        decision=review[c['clip_id']]
                        assert r['object_id'] in decision['reliable_object_ids']+decision['reliable_ee_ids']
                        eligible_by_frame.setdefault(r['frame_idx'],[]).append(m)
                if r['human_confirmed']:assert key in sm
                if 1100<=r['object_id']<=1102:assert not r['training_eligible']
        roi=np.load(release/'task_roi.npz',allow_pickle=False);rr=roi['roi'];kk=roi['known']
        assert rr.shape==kk.shape==(n,h,w) and rr.dtype==kk.dtype==bool
        assert not (rr&~kk).any() and np.array_equal(roi['frame_idx'],data['frame_idx']) and np.array_equal(roi['timestamp'],c['timestamps'])
        fm=json.loads((release/'frames.json').read_text());assert len(fm)==n
        for i,f in enumerate(fm):
            assert int(rr[i].sum())==f['roi_pixels'] and int(kk[i].sum())==f['known_pixels']
            foreground=np.zeros((h,w),np.uint8)
            for m in eligible_by_frame.get(c['start']+i,[]):foreground|=m.astype(np.uint8)
            radius=max(1,round(4*w/640))
            expected=cv2.dilate(foreground,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)
            assert np.array_equal(rr[i],expected)
            expected_known=expected if f['unresolved_object_ids'] else np.ones((h,w),bool)
            assert np.array_equal(kk[i],expected_known)
        cap=cv2.VideoCapture(str(release/'qa.mp4'));decoded=0
        while cap.grab():decoded+=1
        cap.release();assert decoded==n
        entry=dict(clip_id=c['clip_id'],frames=n,rows=len(seen),positive_rows=positive,unknown_rows=unknown,
            reliable_roi_frames=int(rr.reshape(n,-1).any(1).sum()),repeated_camera_timestamps=int((np.diff(c['timestamps'])==0).sum()),
            native_binding=c['native_binding']['status'])
        report.append(entry);totals.update({k:v for k,v in entry.items() if isinstance(v,int)})
    assert totals['frames']==3456
    write_json(ROOT/'independent_validation.json',dict(status='PASS',manifest_sha256=sha(ROOT/'manifest.json'),
        validator_sha256=sha(__file__),clips=report,totals=dict(totals),
        limitations=['Engineering and source correspondence validated; semantic confidence comes from recorded sampled assistant review.',
            'Native LingBot/FastWAM latent/action binding not performed; no calibration or training run.']))
    print('PASS',dict(totals),flush=True)


if __name__=='__main__':cv2.setNumThreads(1);main()
