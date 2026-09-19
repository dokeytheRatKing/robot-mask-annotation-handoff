"""Bounded full-episode assistant recovery, with immutable baselines and review queues."""
import argparse
from collections import Counter, defaultdict
import gc
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from annotate import PROJECT, BASE, sha, write_json
from data import frames, scan
from masks import decode, record
from recovery import diagnostics, Settings
from seed_bank import SeedBank, accepted_rows
from seed_bank_transfer import MODEL, WEIGHTS, candidates, choose, old_rows, original_frame
from seed_bank_transfer_qa import Video, labelled

DEFAULT = PROJECT/'annotations/full_episode_reentry_20260918'
PARENT = PROJECT/'annotations/seed_admission_pilot_20260918'
BANK = PROJECT/'annotations/identity_seed_bank_20260918_curated'
EPISODES = ('episode_001662', 'episode_002503')


def load(root):
    cfg = json.loads((root/'config.json').read_text())
    for name, digest in cfg['code'].items():
        assert sha(BASE/name) == digest, name
    assert sha(BANK/'bank.json') == cfg['bank_sha256']
    return cfg


def rows_at(path):
    result = defaultdict(list)
    with gzip.open(path, 'rt') as f:
        for r in map(json.loads, f):
            result[r['frame_idx']].append(r)
    return result


def temporal_requests(rows, object_ids, cooldown=90):
    """One bounded request per object/reason interval; visibility is never inferred."""
    last = {}; result = []
    for idx in sorted(rows):
        for r in rows[idx]:
            reasons = set(r['suspicious_flags'])-{'automatic_draft_not_gt'}
            if r['mask'] is None:
                reasons.add('unseeded_or_cleared')
            if idx % 120 == 0:
                reasons.add('periodic_identity_check')
            for reason in sorted(reasons):
                key = r['object_id'], reason
                if key in last and idx-last[key] < cooldown:
                    continue
                last[key] = idx
                result.append(dict(frame_idx=idx, object_id=r['object_id'], reason=reason))
    assert all(r['object_id'] in object_ids for r in result)
    return result


def merge_requests(requests, frame_count, window=30):
    groups = defaultdict(list)
    for r in requests:
        groups[r['frame_idx']//window].append(r)
    result = []
    for items in groups.values():
        idx = min(r['frame_idx'] for r in items)
        result.append(dict(frame_idx=idx, object_ids=sorted({r['object_id'] for r in items}),
            reasons=sorted({r['reason'] for r in items}), triggers=items,
            context_frames=sorted({max(0, idx-15), idx, min(frame_count-1, idx+15)}),
            status='pending_assistant_review'))
    return sorted(result, key=lambda r:r['frame_idx'])


def prepare(a):
    a.root.mkdir(exist_ok=False)
    eps = {e['episode_id']:e for e in scan(PROJECT/'datasets/lerobot/astribot_full_v21_rgb_h264')}
    mapping = json.loads((BASE/'config/task_objects.json').read_text())
    pcfg = json.loads((PARENT/'run_config.json').read_text())
    parent_cases = {c['case_id']:c for c in pcfg['cases']}
    source = PARENT/'seeds_r2/seeds.json'
    review = json.loads((PARENT/'seeds_r2/review.json').read_text())
    assert review['seed_sha256'] == sha(source)
    prior = json.loads(source.read_text())
    cases = []
    for ep in EPISODES:
        for cam in ('head', 'left_wrist', 'right_wrist'):
            case = dict(case_id=f'{ep}_{cam}', episode=eps[ep], camera=cam,
                        object_ids=mapping[str(eps[ep]['task_id'])])
            folder = a.root/case['case_id'];folder.mkdir();cache=folder/'frames';cache.mkdir()
            timestamps=[];digest=[];tiles=[]
            for idx,ts,image in frames(eps[ep],cam):
                path=cache/f'{idx:05d}.jpg'
                assert cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,98])
                timestamps.append(ts);digest.append(sha(path))
                if idx%120==0 or idx==eps[ep]['frames']-1:
                    tile=cv2.resize(image,(426,240));tile=np.pad(tile,((24,0),(0,0),(0,0)))
                    cv2.putText(tile,f'{cam} frame {idx}',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    tiles.append(tile)
            while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(folder/'rgb_overview.jpg'),np.concatenate([np.concatenate(tiles[i:i+3],1) for i in range(0,len(tiles),3)],0))
            write_json(folder/'frames.json',dict(timestamps=timestamps,jpeg_sha256=digest,
                shape=image.shape,source_video_sha256=sha(eps[ep]['cameras'][cam]['video'])))
            imported=[]
            for r in prior:
                c=parent_cases[r['case_id']]
                if c['episode']['episode_id']==ep and c['camera']==cam:
                    # Keep scored kettle frames at least 30 frames from any seed.
                    if ep=='episode_001662':continue
                    imported.append(dict(r,case_id=case['case_id'],source_sha256=sha(source),
                                         kind='prior_assistant_review'))
            write_json(folder/'prior_seeds.json',imported)
            cases.append(case)
    cfg=dict(cases=cases,bank_sha256=sha(BANK/'bank.json'),models={'sam2':sha(WEIGHTS)},
        code={n:sha(BASE/n) for n in ('full_episode_reentry.py','seed_bank_transfer.py','seed_bank.py','identity_guard.py','masks.py','recovery.py')},
        parent_seed_sha256=sha(source),human_confirmed=False,robot_subtraction=False,
        policy='Full dense causal propagation. Same initial bank candidates and prior assistant seeds in both arms; only hash-reviewed assistant events added in recovery. No accepted GT used as seeds.',
        limitations=['Two development episodes, not population accuracy.',
                    'Candidate confidence and temporal consistency do not establish identity.',
                    'No automatic GPT service; queued events need actual active-session inspection.'])
    write_json(a.root/'config.json',cfg)
    print([(c['case_id'],c['episode']['frames']) for c in cases],flush=True)


def initial(a):
    from detector import Detector
    from identity_guard import Appearance
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=load(a.root);objects={o['object_id']:o for o in json.loads((BASE/'config/objects.json').read_text())}
    detector=Detector(str(PROJECT/'third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
        str(PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'),str(PROJECT/'models/groundingdino/bert-base-uncased'),objects)
    segmenter=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    encoder=Appearance();bank=SeedBank(BANK)
    for c in cfg['cases']:
        image=original_frame(c,0)
        pool=candidates(image,c['object_ids'],c['episode']['episode_id'],c['camera'],0,detector,segmenter,encoder,bank)
        payload=[]
        for p in pool:
            row={k:v for k,v in p.items() if k not in ('binary','rgb','cutout')}
            payload.append(dict(row,**record(p['binary']),frame_idx=0))
        path=a.root/c['case_id']/'initial_candidates.json';assert not path.exists();write_json(path,payload)
        print('INITIAL',c['case_id'],len(payload),flush=True)


def inference(a):
    from sam2.build_sam import build_sam2_video_predictor
    cfg=load(a.root);objects={o['object_id']:o for o in json.loads((BASE/'config/objects.json').read_text())}
    additions=[];review_hash=None
    if a.mode=='recovery':
        path=a.root/a.revision/'seeds.json';review_path=path.parent/'review.json'
        review=json.loads(review_path.read_text())
        assert review['decision']=='use_as_assisted_seeds' and review['seed_sha256']==sha(path)
        additions=json.loads(path.read_text());review_hash=sha(review_path)
    predictor=build_sam2_video_predictor(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
    for ci,c in enumerate(cfg['cases']):
        if ci%a.shards!=a.shard:continue
        folder=a.root/c['case_id'];out=folder/a.mode;out.mkdir(exist_ok=False)
        fm=json.loads((folder/'frames.json').read_text());n=c['episode']['frames'];assert len(fm['timestamps'])==n
        seeds={};pool=[dict(r,binary=decode(r['mask'])) for r in json.loads((folder/'initial_candidates.json').read_text())]
        for oid in c['object_ids']:
            chosen,_=choose([p for p in pool if p['object_id']==oid],'bank_rank',True)
            if chosen:
                seeds[0,oid]=dict(chosen,kind='automatic_bank_initialization',prompt={'status':'visible'},
                    source_sha256=sha(folder/'initial_candidates.json'))
        for r in [*json.loads((folder/'prior_seeds.json').read_text()),*additions]:
            if r['case_id']==c['case_id']:
                seeds[r['frame_idx'],r['object_id']]=r
        boundaries=sorted({0,n,*range(0,n,30),*(idx for idx,_ in seeds)})
        robot,robot_source=old_rows(c['episode'],c['camera'],0,n-1,'robot')
        robot_parts,part_source=old_rows(c['episode'],c['camera'],0,n-1,'robot_parts')
        tracks={oid:dict(active=False,seed=None) for oid in c['object_ids']};previous={};events=[]
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),gzip.open(out/'objects.jsonl.gz','wt') as f:
            state=predictor.init_state(str(folder/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            for idx,end in zip(boundaries[:-1],boundaries[1:]):
                for oid,t in tracks.items():
                    seed=seeds.get((idx,oid))
                    if seed is None:continue
                    if t['active']:predictor.remove_object(state,oid,strict=True,need_output=False)
                    t.update(active=False,seed=None)
                    action='clear_unknown'
                    if seed['prompt']['status']=='visible':
                        predictor.add_new_mask(state,idx,oid,decode(seed['mask']))
                        t.update(active=True,seed=dict(frame_idx=idx,kind=seed['kind'],
                            source_sha256=seed['source_sha256'],human_confirmed=False,
                            candidate_id=seed.get('candidate_id'),retrieval=seed.get('retrieval')))
                        action='seed'
                    previous.pop(oid,None)
                    events.append(dict(frame_idx=idx,object_id=oid,action=action,kind=seed['kind']))
                predictions={}
                if any(t['active'] for t in tracks.values()):
                    for actual,oids,logits in predictor.propagate_in_video(state,start_frame_idx=idx,max_frame_num_to_track=end-idx-1):
                        assert idx<=actual<end;predictions[actual]={}
                        for oid,m in zip(oids,(logits[:,0]>0).cpu().numpy()):
                            cache=state['output_dict_per_obj'][state['obj_id_to_idx'][oid]]
                            val=cache['cond_frame_outputs'].get(actual)
                            if val is None:val=cache['non_cond_frame_outputs'][actual]
                            predictions[actual][oid]=(m,float(val['object_score_logits'].float().sigmoid().reshape(-1)[0]))
                    assert set(predictions)==set(range(idx,end))
                for actual in range(idx,end):
                    binary={oid:pair[0] for oid,pair in predictions.get(actual,{}).items()}
                    flags=diagnostics(binary,previous,decode(robot[actual][0]['mask']),Settings())
                    for oid,t in tracks.items():
                        r=dict(episode_id=c['episode']['episode_id'],task_id=c['episode']['task_id'],camera=c['camera'],
                            frame_idx=actual,timestamp=fm['timestamps'][actual],object_id=oid,class_name=objects[oid]['class_name'],
                            image_width=fm['shape'][1],image_height=fm['shape'][0],human_confirmed=False,mode=a.mode,
                            provenance=t['seed'],seed_frame=t['seed']['frame_idx'] if t['seed'] else None,
                            robot_subtraction=False,suspicious_flags=flags.get(oid,[]))
                        if oid in binary:
                            m,prob=predictions[actual][oid];r.update(**record(m),confidence=prob,
                                visibility='predicted_visible' if m.any() else 'unknown')
                        else:r.update(mask=None,bbox_xyxy=None,mask_area=None,visible=None,confidence=None,visibility='unknown')
                        f.write(json.dumps(r,allow_nan=False)+'\n')
                    previous=binary
                print(a.mode,c['case_id'],end,n,flush=True)
            del state
        write_json(out/'events.json',events)
        write_json(out/'complete.json',dict(seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,frames=n,
            config_sha256=sha(a.root/'config.json'),initial_sha256=sha(folder/'initial_candidates.json'),
            prior_sha256=sha(folder/'prior_seeds.json'),review_sha256=review_hash,
            objects_sha256=sha(out/'objects.jsonl.gz'),events_sha256=sha(out/'events.json'),
            robot_source=robot_source,robot_parts_source=part_source))
        del robot,robot_parts;gc.collect();print('COMPLETE',a.mode,c['case_id'],flush=True)


def queue(a):
    cfg=load(a.root);all_requests=[]
    for c in cfg['cases']:
        folder=a.root/c['case_id'];source=folder/a.mode/'objects.jsonl.gz'
        rows=rows_at(source);requests=temporal_requests(rows,c['object_ids'])
        grouped=merge_requests(requests,c['episode']['frames']);out=folder/f'{a.mode}_queue';out.mkdir(exist_ok=False)
        for q in grouped:
            cells=[]
            for idx in q['context_frames']:
                image=cv2.imread(str(folder/'frames'/f'{idx:05d}.jpg'))
                for layer,title in [([],f'{c["camera"]} RGB {idx}'),(rows[idx],f'{a.mode} {idx}')]:
                    rgb,legend=labelled(image,layer,title)
                    legend=np.pad(legend,((0,len(c['object_ids'])*16-legend.shape[0]),(0,0),(0,0)))
                    cells.append(np.concatenate([rgb,legend],0))
            # Each temporal position has RGB above prediction.
            canvas=np.concatenate([np.concatenate(cells[0::2],1),np.concatenate(cells[1::2],1)],0)
            path=out/f'{q["frame_idx"]:05d}.jpg';cv2.imwrite(str(path),canvas)
            q.update(case_id=c['case_id'],image=str(path.relative_to(a.root)),image_sha256=sha(path),
                     prediction_sha256=sha(source))
            all_requests.append(q)
        print('QUEUE',c['case_id'],len(requests),len(grouped),flush=True)
    path=a.root/f'{a.mode}_review_queue.json';assert not path.exists()
    write_json(path,all_requests)


def seed_proposals(a):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg=load(a.root);cases={c['case_id']:c for c in cfg['cases']}
    prompts=json.loads(a.prompts.read_text());dest=a.root/a.revision;dest.mkdir(exist_ok=False)
    predictor=SAM2ImagePredictor(build_sam2(MODEL,str(WEIGHTS),device='cuda',apply_postprocessing=False))
    result=[];previous=None;keys=set()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for p in prompts:
            c=cases[p['case_id']];key=(p['case_id'],p['frame_idx']);oid=p['object_id']
            assert (*key,oid) not in keys;keys.add((*key,oid))
            assert oid in c['object_ids'] and 0<=p['frame_idx']<c['episode']['frames']
            assert p['status'] in ('visible','not_visible','uncertain')
            assert not (c['episode']['episode_id']=='episode_001662' and abs(p['frame_idx']-149)<30), 'Reserved scoring gap'
            if key!=previous:
                image=original_frame(c,p['frame_idx']);predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));previous=key
                cv2.imwrite(str(dest/f'{key[0]}_{key[1]}_rgb.jpg'),image)
            mask=np.zeros(image.shape[:2],bool);scores=[]
            if p['status']=='visible':
                for comp in [p,*p.get('components',[])]:
                    m,values,_=predictor.predict(box=np.array(comp['box']),point_coords=np.array(comp['points']),
                        point_labels=np.array(comp['point_labels']),multimask_output=False)
                    mask|=m[0].astype(bool);scores.append(float(values[0]))
                assert mask.any(),p
            row=dict(case_id=p['case_id'],frame_idx=p['frame_idx'],object_id=oid,prompt=p,**record(mask),
                sam_predicted_iou=scores,human_confirmed=False,kind='assistant_context_reseed',source_sha256=sha(a.prompts))
            result.append(row)
            rgb,legend=labelled(image,[dict(row,class_name=str(oid),confidence=None)],f'{key[0]} {key[1]} id{oid} {p["status"]}')
            cv2.imwrite(str(dest/f'{key[0]}_{key[1]}_{oid}.jpg'),np.concatenate([rgb,legend],0))
    write_json(dest/'seeds.json',result)
    write_json(dest/'manifest.json',dict(prompts=prompts,prompt_sha256=sha(a.prompts),seed_sha256=sha(dest/'seeds.json'),
        config_sha256=sha(a.root/'config.json'),human_confirmed=False))
    print('SEEDS',len(result),len({(p['case_id'],p['frame_idx']) for p in prompts}),flush=True)


def finish(a):
    cfg=load(a.root);qa=a.root/'qa';qa.mkdir(exist_ok=False);results=[];validation=[]
    labels=[r for r in accepted_rows() if r['episode_id'] in EPISODES and r['object_id']<1000 and not r.get('exclude_from_metrics')]
    for c in cfg['cases']:
        folder=a.root/c['case_id'];fm=json.loads((folder/'frames.json').read_text());layers={};n=c['episode']['frames']
        for mode in ('baseline','recovery'):
            meta=json.loads((folder/mode/'complete.json').read_text());path=folder/mode/'objects.jsonl.gz'
            assert sha(path)==meta['objects_sha256'] and sha(folder/mode/'events.json')==meta['events_sha256']
            assert meta['config_sha256']==sha(a.root/'config.json')
            for key in ('robot_source','robot_parts_source'):
                assert sha(meta[key]['path'])==meta[key]['sha256']
            rows=rows_at(path);keys=[]
            for idx,rr in rows.items():
                for r in rr:
                    keys.append((idx,r['object_id']));assert r['timestamp']==fm['timestamps'][idx]
                    assert r['human_confirmed'] is False and r['robot_subtraction'] is False
                    if r['mask'] is not None:
                        m=decode(r['mask']);geom=record(m);assert m.shape==tuple(fm['shape'][:2])
                        assert geom['bbox_xyxy']==r['bbox_xyxy'] and geom['mask_area']==r['mask_area']
                        assert r['seed_frame']<=idx
            assert len(keys)==len(set(keys))==n*len(c['object_ids'])
            assert set(keys)=={(idx,oid) for idx in range(n) for oid in c['object_ids']}
            layers[mode]=rows
            for gt in labels:
                if gt['episode_id']!=c['episode']['episode_id'] or gt['camera']!=c['camera']:continue
                idx=gt['frame_idx'];r=next(x for x in rows[idx] if x['object_id']==gt['object_id'])
                assert np.array_equal(original_frame(c,idx),cv2.imread(gt['rgb_path']))
                truth=decode(gt['mask']);pred=decode(r['mask']) if r['mask'] else np.zeros_like(truth)
                assert r['seed_frame'] is None or idx-r['seed_frame']>=30
                union=(truth|pred).sum();total=truth.sum()+pred.sum()
                results.append(dict(case_id=c['case_id'],mode=mode,frame_idx=idx,object_id=gt['object_id'],
                    gt_visible=bool(truth.any()),pred_visible=bool(pred.any()),seed_frame=r['seed_frame'],
                    iou=float((truth&pred).sum()/union) if union else None,
                    dice=float(2*(truth&pred).sum()/total) if total else None))
        robot,robot_source=old_rows(c['episode'],c['camera'],0,n-1,'robot_parts')
        out=qa/c['case_id'];out.mkdir();writer=Video(out/'comparison.mp4',c['episode']['fps'])
        contacts=set(range(0,n,120))|{n-1}
        if c['episode']['episode_id']=='episode_002503':contacts|={588,608,618,638,668}
        for idx,ts,image in frames(c['episode'],c['camera']):
            cells=[]
            for rows,title in [([],f'{c["camera"]} RGB {idx}'),(layers['baseline'][idx],'Baseline'),
                               (layers['recovery'][idx],'Assistant recovery'),(robot[idx],'Robot parts unchanged, not accepted')]:
                rgb,legend=labelled(image,rows,title);legend=np.pad(legend,((0,5*16-legend.shape[0]),(0,0),(0,0)))
                cells.append(np.concatenate([rgb,legend],0))
            canvas=np.concatenate([np.concatenate(cells[:2],1),np.concatenate(cells[2:],1)],0);writer.add(canvas)
            if idx in contacts:cv2.imwrite(str(out/f'{idx:05d}.jpg'),canvas)
        writer.close();cap=cv2.VideoCapture(str(out/'comparison.mp4'));decoded=0
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==writer.shape;decoded+=1
        cap.release();assert decoded==n
        validation.append(dict(case_id=c['case_id'],frames=n,rows_per_mode=n*len(c['object_ids']),
            video_sha256=sha(out/'comparison.mp4'),robot_source=robot_source))
        print('VALIDATED',c['case_id'],n,flush=True)
    write_json(a.root/'scoring.json',dict(rows=results,warning='Only two task1 accepted labels; no annotated temporal GT for new task24. No population accuracy/re-entry rate.'))
    write_json(a.root/'validation.json',dict(status='PASS',semantic_acceptance=False,cases=validation,
        camera_frames=sum(r['frames'] for r in validation),rows=sum(r['rows_per_mode']*2 for r in validation)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','initial','run','queue','seeds','finish'])
    p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--mode',choices=['baseline','recovery'],default='baseline')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--prompts',type=Path);p.add_argument('--revision',default='seeds_r1')
    a=p.parse_args();assert 0<=a.shard<a.shards
    torch.set_num_threads(3);cv2.setNumThreads(1);torch.manual_seed(20260918)
    {'prepare':prepare,'initial':initial,'run':inference,'queue':queue,'seeds':seed_proposals,'finish':finish}[a.phase](a)
