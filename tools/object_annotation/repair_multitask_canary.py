"""Post-score local recovery candidates; fixed benchmark outputs remain immutable."""
import argparse
import copy
import gzip
import json
import cv2
import numpy as np
import torch
from annotate import BASE, sha, write_json
from full_episode_reentry import rows_at
from masks import decode, record
from multitask_reseed_canary import ROOT, load, read_image, propagate
from seed_bank_transfer import MODEL, WEIGHTS
from seed_bank_transfer_qa import labelled, Video

SPEC=BASE/'config/multitask_canary_repair.json'


def inspect():
    cfg=load(ROOT);out=ROOT/'recovery';out.mkdir(exist_ok=False)
    wanted={'T13_episode_005300_right_wrist':[265,270,275,285],
        'T27_episode_002979_right_wrist':[754,758,762,766,770,775,785,790,795,800]}
    for cid,indices in wanted.items():
        c=next(c for c in cfg['cases'] if c['case_id']==cid);rows=rows_at(ROOT/cid/'predictions/multi_seed.jsonl.gz')
        for part in range(0,len(indices),4):
            tiles=[]
            for idx in indices[part:part+4]:
                image=read_image(ROOT,c,idx);rgb,_=labelled(image,[],f'{cid} frame {idx}')
                overlay,_=labelled(image,rows[idx],'Frozen multi_seed')
                tiles.append(np.concatenate([rgb,overlay],1))
            cv2.imwrite(str(out/f'{cid}_{part//4}.jpg'),np.concatenate(tiles,0))


def seeds():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=load(ROOT);spec=json.loads(SPEC.read_text());out=ROOT/'recovery';rows=[]
    predictor=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for event in spec['events']:
            c=next(c for c in cfg['cases'] if c['case_id']==event['case_id']);idx=event['start']
            image=read_image(ROOT,c,idx);h,w=image.shape[:2];m=np.zeros((h,w),bool)
            p=event['prompt'];score=None
            if p['status']=='visible':
                predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                mm,ss,_=predictor.predict(box=np.asarray(p['box']),point_coords=np.asarray(p['points']),
                    point_labels=np.asarray(p['point_labels']),multimask_output=False)
                m=mm[0].astype(bool);score=float(ss[0]);assert m.any()
            r=dict(event,frame_idx=idx,**record(m),confidence=None,class_name=str(event['object_id']),sam_predicted_iou=score,human_confirmed=False)
            rows.append(r);rgb,_=labelled(image,[],f'{event["case_id"]} {idx}')
            mask,_=labelled(image,[r],f'Post-score repair seed id{r["object_id"]} {p["status"]}')
            cv2.imwrite(str(out/f'seed_{len(rows)}.jpg'),np.concatenate([rgb,mask],1))
    write_json(out/'seeds.json',rows)
    write_json(out/'provenance.json',dict(spec_sha256=sha(SPEC),seed_sha256=sha(out/'seeds.json'),
        script_sha256=sha(__file__),fixed_metrics_sha256=sha(ROOT/'metrics.json'),status='post_score_development_not_new_test',
        causal=True,human_confirmed=False))


def run():
    from sam2.build_sam import build_sam2_video_predictor
    cfg=load(ROOT);out=ROOT/'recovery';events=json.loads((out/'seeds.json').read_text())
    provenance=json.loads((out/'provenance.json').read_text());review=json.loads((out/'seed_review.json').read_text())
    assert provenance['seed_sha256']==sha(out/'seeds.json')==review['seed_sha256']
    assert provenance['script_sha256']==sha(__file__) and provenance['spec_sha256']==sha(SPEC)
    assert review['decision']=='use_as_assisted_seeds'
    predictor=build_sam2_video_predictor(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    outputs=[]
    for cid in sorted({e['case_id'] for e in events}):
        c=next(c for c in cfg['cases'] if c['case_id']==cid);source=ROOT/cid/'predictions/multi_seed.jsonl.gz'
        original=rows_at(source);merged=copy.deepcopy(original);targets=set()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for e in [e for e in events if e['case_id']==cid]:
                pred=propagate(predictor,ROOT,c,[e],e['start'],e['end'])
                for idx in range(e['start'],e['end']+1):
                    for r in merged[idx]:
                        if r['object_id']!=e['object_id']:continue
                        assert (idx,r['object_id']) not in targets;targets.add((idx,r['object_id']))
                        r.update(mask=None,mask_area=None,bbox_xyxy=None,visible=None,visibility='unknown',confidence=None,
                            mask_status='unknown',mode='post_score_recovery',seed_frame=e['start'],suspicious_flags=['post_score_assisted_candidate_not_gt'],
                            provenance=dict(kind='post_score_local_recovery',source_sha256=sha(source),seed_sha256=sha(out/'seeds.json'),causal=True,event_frame=e['start']))
                        if idx in pred and e['object_id'] in pred[idx] and pred[idx][e['object_id']][0].any():
                            m,prob=pred[idx][e['object_id']];r.update(**record(m),confidence=prob,visibility='predicted_visible')
        path=out/f'{cid}.jsonl.gz';assert not path.exists();unchanged=0
        with gzip.open(path,'wt',encoding='utf-8') as f:
            for idx,rows in merged.items():
                for r in rows:
                    before=next(x for x in original[idx] if x['object_id']==r['object_id'])
                    if (idx,r['object_id']) not in targets:assert r==before;unchanged+=1
                    assert r['timestamp']==c['timestamps'][str(idx)]
                    if r['mask'] is not None:
                        m=decode(r['mask']);geo=record(m);assert geo['bbox_xyxy']==r['bbox_xyxy'] and geo['mask_area']==r['mask_area']
                    f.write(json.dumps(r,allow_nan=False)+'\n')
        q=out/cid;q.mkdir();writer=Video(q/'comparison.mp4',30)
        checks=sorted(set([e['start'] for e in events if e['case_id']==cid]+[c['eval_frame'],c['end']]))
        for idx,rs in merged.items():
            image=read_image(ROOT,c,idx);cells=[]
            for rr,title in [([],f'{cid} {idx} RGB'),(original[idx],'Fixed multi_seed'),(rs,'Post-score recovery candidate')]:
                rgb,legend=labelled(image,rr,title);legend=np.pad(legend,((0,80-legend.shape[0]),(0,0),(0,0)));cells.append(np.concatenate([rgb,legend],0))
            canvas=np.concatenate(cells,1);writer.add(canvas)
            if idx in checks:cv2.imwrite(str(q/f'{idx:05d}.jpg'),canvas)
        writer.close();cap=cv2.VideoCapture(str(q/'comparison.mp4'));n=0
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==writer.shape;n+=1
        cap.release();assert n==c['end']-c['start']+1
        outputs.append(dict(case_id=cid,modified_rows=len(targets),unchanged_rows=unchanged,frames=n,path=str(path),sha256=sha(path)))
    assert provenance['fixed_metrics_sha256']==sha(ROOT/'metrics.json')
    write_json(out/'validation.json',dict(status='PASS',outputs=outputs,post_score_development=True,
        no_new_test_accuracy_claim=True,robot_layers_unchanged=True,semantic_acceptance=False))
    print(json.dumps(outputs,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['inspect','seeds','run']);a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20260919);globals()[a.phase]()
