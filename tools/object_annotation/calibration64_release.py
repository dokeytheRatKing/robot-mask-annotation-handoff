"""Release reviewed clip masks, reliable ROI, data contract and offline QA."""
import argparse
import ast
from collections import Counter
import copy
import csv
import gzip
import html
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from annotate import BASE, PROJECT, sha, write_json
from calibration64 import ROOT, load, read
from full_episode_reentry import rows_at
from masks import decode, encode, record
from seed_bank_transfer_qa import Video, labelled


def finalize(a):
    cfg=load(a.root);review=read(a.root,'review.json');items=review['clips']
    assert len(items)==64 and set(items)=={c['clip_id'] for c in cfg['clips']}
    summary=[];index=[]
    for ci,c in enumerate(cfg['clips']):
        if ci%a.shards!=a.shard:continue
        out=a.root/'clips'/c['clip_id'];decision=items[c['clip_id']]
        c=copy.deepcopy(c)
        if 'object_ids_override' in decision:c['object_ids']=decision['object_ids_override']
        assert decision['status']=='reviewed_for_bounded_calibration'
        source=out/'draft';obj=rows_at(source/'objects.jsonl.gz');rob=rows_at(source/'robot_parts.jsonl.gz')
        # Optional explicit local fixes are versioned before release.
        if decision.get('repair_objects'):
            path=a.root/decision['repair_objects'];assert sha(path)==decision['repair_objects_sha256'];obj=rows_at(path)
        if decision.get('repair_robot'):
            path=a.root/decision['repair_robot'];assert sha(path)==decision['repair_robot_sha256'];rob=rows_at(path)
        dest=out/'release';dest.mkdir(exist_ok=False);h,w=c['shape'][:2]
        rois=[];known=[];manifest=[];count=Counter();writer=Video(dest/'qa.mp4',30)
        exact=read(out,'accepted_seeds.json');seedmap={(r['frame_idx'],r['object_id']):r for r in exact}
        allow=set(decision['reliable_object_ids'])|set(decision['reliable_ee_ids'])
        absent=set(decision.get('reviewed_absent_ids',[]));exclude=set(decision.get('exclude_ids',[]))
        incomplete=set(decision.get('incomplete_object_ids',[]))
        assert allow<=set(c['object_ids']+c['robot_part_ids']) and not any(1100<=x<=1102 for x in allow)
        assert not (allow&exclude) and not (allow&absent)
        handles={layer:gzip.open(dest/f'{layer}.jsonl.gz','wt',encoding='utf-8') for layer in ['objects','robot_parts']}
        for idx in range(c['start'],c['end']+1):
            local=idx-c['start'];image=cv2.imread(str(out/'rgb'/f'{local:05d}.png'))
            assert sha(out/'rgb'/f'{local:05d}.png')==c['rgb_sha256'][local]
            foreground=np.zeros((h,w),bool);unresolved=[];finalobjects=[];finalrobot=[]
            for layer,source_rows,acc in [('objects',obj.get(idx,[]),finalobjects),('robot_parts',rob.get(idx,[]),finalrobot)]:
                for raw in source_rows:
                    r=copy.deepcopy(raw);oid=r['object_id'];is_seed=(idx,oid) in seedmap
                    assert r['timestamp']==c['timestamps'][local]
                    r['training_eligible']=False;r['review_status']='assistant_clip_review'
                    r['review_sha256']=sha(a.root/'review.json')
                    interval=next((e for e in decision.get('frame_exclusions',[]) if e['object_id']==oid and e['start']<=idx<=e['end']),None)
                    if interval:
                        assert not (is_seed and seedmap[(idx,oid)]['visible'] is True),'Accepted positive must remain intact'
                        r.update(mask=None,bbox_xyxy=None,mask_area=None,visible=None,visibility='unknown',confidence=None,
                            mask_status='unknown',human_confirmed=False)
                        r['suspicious_flags']=sorted(set(r.get('suspicious_flags',[])+['reviewed_interval_unknown',interval['reason']]))
                        r['review_exclusion']=interval
                        if is_seed and seedmap[(idx,oid)]['visible'] is False:
                            r['suspicious_flags'].append('accepted_negative_discrepancy_unknown')
                        if oid<1000:unresolved.append(oid)
                    # Never reinterpret a non-empty, exact accepted seed as absent.
                    elif oid in absent and not (is_seed and r['visible'] is True):
                        r.update(record(np.zeros((h,w),bool)))
                        r.update(visibility='reviewed_likely_absent',confidence=None,
                            mask_status='accepted_negative' if is_seed else 'assisted_negative')
                    elif oid in exclude:
                        # Retain predicted geometry separately, excluded from all supervision.
                        r['review_status']='unreliable_excluded';r['suspicious_flags']=list(set(r.get('suspicious_flags',[])+['excluded_from_roi']))
                        if oid<1000:unresolved.append(oid)
                    elif oid in allow and r['mask'] is not None and r['visible']:
                        m=decode(r['mask']);assert m.shape==(h,w)
                        if m.any():foreground|=m;r['training_eligible']=True
                        else:unresolved.append(oid)
                    elif oid<1000 and oid not in absent and not (is_seed and r['visible'] is False):unresolved.append(oid)
                    if oid in incomplete:
                        unresolved.append(oid);r['suspicious_flags']=sorted(set(r.get('suspicious_flags',[])+['partial_mask_unknown_background']))
                    if r['mask'] is not None:
                        m=decode(r['mask']);geom=record(m)
                        assert geom['bbox_xyxy']==r['bbox_xyxy'] and geom['mask_area']==r['mask_area']
                    if is_seed:
                        gt=seedmap[(idx,oid)]
                        if gt['mask'] is not None and oid not in exclude:
                            if 'accepted_negative_discrepancy_unknown' in r.get('suspicious_flags',[]):
                                assert gt['visible'] is False and r['mask'] is None and not r['human_confirmed']
                            else:assert np.array_equal(decode(gt['mask']),decode(r['mask']))
                    handles[layer].write(json.dumps(r,allow_nan=False)+'\n');acc.append(r);count['rows']+=1
                    count['eligible_rows']+=bool(r['training_eligible'])
            radius=max(1,round(4*w/640));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
            roi=cv2.dilate(foreground.astype('uint8'),kernel).astype(bool)
            # Complete object identity inventory reviewed -> background known.
            # Otherwise only reliable ROI pixels known; no negative hallucination.
            k=np.ones((h,w),bool) if not unresolved else roi.copy()
            rois.append(roi);known.append(k)
            frame=dict(local_frame_idx=local,frame_idx=idx,timestamp=c['timestamps'][local],
                prediction_frame=idx in c['prediction_frames'],roi_pixels=int(roi.sum()),known_pixels=int(k.sum()),
                unresolved_object_ids=sorted(set(unresolved)),has_reliable_roi=bool(roi.any()),margin_radius_pixels=radius)
            manifest.append(frame);count['frames_with_roi']+=int(roi.any());count['frames_with_unresolved_objects']+=bool(unresolved)
            cells=[]
            overlay=image.copy();overlay[roi]=(overlay[roi]*.55+np.array([30,220,245])*.45).astype('uint8')
            for rows,label in [([],f'{c["clip_id"][:8]} {c["camera"]} RGB {idx}'),(finalobjects,'Objects'),(finalrobot,'Robot separate; arms excluded from ROI')]:
                im,legend=labelled(image,rows,label);legend=np.pad(legend,((0,max(0,80-legend.shape[0])),(0,0),(0,0)));cells.append(np.concatenate([im,legend],0))
            im,legend=labelled(overlay,[],f'Reliable task ROI {int(roi.sum())}px');legend=np.pad(legend,((0,80-legend.shape[0]),(0,0),(0,0)));cells.append(np.concatenate([im,legend],0))
            canvas=np.concatenate([np.concatenate(cells[:2],1),np.concatenate(cells[2:],1)],0);writer.add(canvas)
            if idx in [c['start'],c['end'],c['anchor_frame']]:cv2.imwrite(str(dest/f'qa_{idx:05d}.jpg'),canvas)
        for f in handles.values():f.close()
        writer.close();cap=cv2.VideoCapture(str(dest/'qa.mp4'));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==writer.shape;decoded+=1
        cap.release();assert decoded==c['frame_count']
        np.savez_compressed(dest/'task_roi.npz',roi=np.stack(rois),known=np.stack(known),
            frame_idx=np.arange(c['start'],c['end']+1),timestamp=np.asarray(c['timestamps'],np.float64))
        write_json(dest/'frames.json',manifest)
        write_json(dest/'validation.json',dict(status='PASS',review_sha256=sha(a.root/'review.json'),
            script_sha256=sha(__file__),counts=dict(count),decoded_video_frames=decoded,
            hashes={n:sha(dest/n) for n in ['objects.jsonl.gz','robot_parts.jsonl.gz','task_roi.npz','frames.json','qa.mp4']},
            supervision='Reviewed assisted pseudo-labels; only original seed pixels are human-confirmed. No corpus accuracy claim.'))
        print('RELEASED',c['clip_id'],dict(count),flush=True)


def package(a):
    cfg=load(a.root);splits=read(a.root,'splits.json');report=PROJECT/'docs/action_metric_calibration64_data_report_20260919.md'
    assert report.exists();entries=[];totals=Counter()
    assert read(a.root,'independent_validation.json')['status']=='PASS'
    assert len(splits['C_fit'])==48 and len(splits['C_diag'])==16 and len(splits['H'])==16
    assert not(set(splits['C_fit'])&set(splits['C_diag'])) and set(splits['H'])<=set(splits['C_fit'])
    assert len({c['episode']['episode_id'] for c in cfg['clips']})==64
    for c in cfg['clips']:
        dest=a.root/'clips'/c['clip_id']/'release';v=read(dest,'validation.json');assert v['status']=='PASS'
        for name,h in v['hashes'].items():assert sha(dest/name)==h
        totals.update(v['counts']);totals['frames']+=v['decoded_video_frames']
        entry={k:c[k] for k in ['clip_id','task_id','camera','side','split','in_H','start','end','frame_count',
            'source_horizon_frames','physical_horizon_seconds','object_ids','robot_part_ids','random_seeds','native_binding']}
        entry.update(episode_id=c['episode']['episode_id'],relative_root=f'clips/{c["clip_id"]}',
            mask_version='release',mask_review=read(a.root,'review.json')['clips'][c['clip_id']],
            source_hashes=dict(video=c['source_rgb_sha256'],parquet=c['source_parquet_sha256']),output_hashes=v['hashes'])
        if 'object_ids_override' in entry['mask_review']:entry['object_ids']=entry['mask_review']['object_ids_override']
        entries.append(entry)
    coverage=dict(tasks=dict(Counter(str(c['task_id']) for c in cfg['clips'])),cameras=dict(Counter(c['camera'] for c in cfg['clips'])),
        horizons=dict(Counter(str(c['source_horizon_frames']) for c in cfg['clips'])),sides=dict(Counter(c['side'] for c in cfg['clips'])),
        fit_tasks=sorted({c['task_id'] for c in cfg['clips'] if c['split']=='C_fit'}),
        diagnostic_tasks=sorted({c['task_id'] for c in cfg['clips'] if c['split']=='C_diag'}))
    write_json(a.root/'dataset.json',dict(schema='astribot.calibration64.release.v1',clips=entries,coverage=coverage,
        totals=dict(totals),action=cfg['action'],state=cfg['state'],manifest_sha256=sha(a.root/'manifest.json'),
        review_sha256=sha(a.root/'review.json'),mask_contract=dict(roi='bool N,H,W; task objects + reviewed reliable EE + fixed local margin',
            known='bool N,H,W; unknown ROI=0; preserve separately from known background',
            latent_alignment='Not yet materialized; adapter requires explicit native VAE temporal support and spatial transform.',
            mask_quality='assistant-reviewed pseudo-labels, original accepted seeds retain human confirmation'),
        status='data_prepared_native_model_binding_pending',training_started=False))
    with (a.root/'clips.csv').open('w',encoding='utf-8',newline='') as f:
        keys=['clip_id','episode_id','task_id','camera','side','split','in_H','start','end','frame_count','source_horizon_frames','physical_horizon_seconds']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:e[k] for k in keys} for e in entries)
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>64 clip 校准数据</title>',
        '<style>body{font:16px system-ui;background:#181b1e;color:#eee;max-width:1200px;margin:auto;padding:20px}a{color:#9bd7ff}video,img{width:100%}section{border-top:1px solid #555;margin-top:25px;padding-top:15px}</style>',
        '<h1>固定 64 clip 校准数据</h1><p>48 C_fit / 16 C_diag；H 为 C_fit 的16个clip子集。每个clip来自不同episode，全部在训练划分内。</p>',
        '<p>原始25D state、20D absolute action和三相机物理时间保存；本批每个clip仅标注其指定相机。其余相机原视频路径保留在manifest，没有伪造其它视角mask。</p>',
        '<p>视频：RGB / object masks / separate robot masks / reliable task ROI。仅可靠EE进入ROI，臂不进入。未知保留为unknown。已有人工seed之外都是经助手检查的pseudo-labels。</p>',
        '<p><a href="report.md">报告</a> · <a href="clips.csv">样本清单</a> · <a href="dataset.json">数据契约</a></p>']
    for c in cfg['clips']:
        cid=c['clip_id'];page.append(f'<section><h2>{html.escape(cid)}</h2><p>{c["split"]}{" / H" if c["in_H"] else ""}；源帧 {c["start"]}–{c["end"]}；预测窗 {c["source_horizon_frames"]} 行 / 实际 {c["physical_horizon_seconds"]:.6f} s；QA按30 fps播放</p><img loading="lazy" src="clips/{cid}/release/qa_{c["anchor_frame"]:05d}.jpg"><video controls preload="none" src="clips/{cid}/release/qa.mp4"></video></section>')
    page.append('</html>');(a.root/'index.html').write_text('\n'.join(page),encoding='utf-8')
    shutil.copyfile(report,a.root/'report.md')
    files={n:a.root/n for n in ['index.html','manifest.json','dataset.json','splits.json','clips.csv','review.json',
        'independent_validation.json','assistant_review_notes.json']};files['report.md']=report
    for folder in ['repairs','repairs_v2']:
        for p in (a.root/folder).rglob('*'):
            if p.is_file():files[str(p.relative_to(a.root))]=p
    for c in cfg['clips']:
        root=a.root/'clips'/c['clip_id']
        for p in root.rglob('*'):
            if p.is_file() and (p.parent.name in ['rgb','release'] or p.name in ['state_action.npz','state_action.parquet','accepted_seeds.json']):
                files[str(p.relative_to(a.root))]=p
    pending=['calibration64.py','calibration64_release.py','calibration64_mask_adapter.py','validate_calibration64.py',
        'repair_calibration64.py','build_calibration64_review.py'];visited=set()
    while pending:
        name=pending.pop()
        if name in visited:continue
        visited.add(name);files['code/'+name]=BASE/name
        for node in ast.walk(ast.parse((BASE/name).read_text())):
            modules=[node.module] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names] if isinstance(node,ast.Import) else []
            for mod in modules:
                local=(mod or '').split('.')[0]+'.py'
                if (BASE/local).is_file() and local not in visited:pending.append(local)
    for name in ['config/calibration64_repairs.json','config/objects.json','config/task_objects.json']:
        files['code/'+name]=BASE/name
    hashes={name:sha(path) for name,path in files.items()};target=PROJECT/'deliverables/astribot_action_metric_calibration64_20260919.zip'
    assert not target.exists()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name,path in files.items():z.write(path,name)
        z.writestr('SHA256SUMS',''.join(f'{h}  {n}\n' for n,h in sorted(hashes.items())))
    import hashlib
    from html.parser import HTMLParser
    class Check(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['src','href']:assert v in files,v
    Check().feed((a.root/'index.html').read_text())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    write_json(a.root/'package.json',dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),verified_files=len(files),coverage=coverage,totals=dict(totals)))
    print(json.dumps(read(a.root,'package.json'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['finalize','package']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();cv2.setNumThreads(1);globals()[a.phase](a)
