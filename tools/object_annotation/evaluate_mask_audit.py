"""Independent human-GT evaluation. Missing GT produces null, never proxy scores."""
import argparse
from collections import defaultdict,Counter
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from annotate import write_json,sha
from masks import decode,bbox

VISIBLE={'visible','partial_occlusion'}
INVISIBLE={'out_of_view','fully_occluded'}


def similarity(pred,gt):
    inter=int((pred & gt).sum());total=int(pred.sum()+gt.sum());union=int((pred|gt).sum())
    return (inter/union if union else None,2*inter/total if total else None)


def pair_score(prediction,label):
    gt=decode(label['mask']);pred=np.zeros_like(gt) if prediction is None or prediction['mask'] is None else decode(prediction['mask'])
    assert pred.shape==gt.shape
    iou,dice=similarity(pred,gt)
    return dict(iou=iou,dice=dice,pred_visible=prediction['visible'] if prediction else None,
        gt_visible=label['visibility'] in VISIBLE,gt_visibility=label['visibility'],pred_area=int(pred.sum()))


def aggregate(entries):
    visible=[e for e in entries if e['gt_visible']]
    invisible=[e for e in entries if not e['gt_visible']]
    out=[e for e in entries if e['gt_visibility']=='out_of_view']
    mean=lambda xs:float(np.mean(xs)) if xs else None
    return dict(labeled_object_frames=len(entries),visible_gt=len(visible),invisible_gt=len(invisible),
        mask_iou=mean([e['iou'] for e in visible]),mask_dice=mean([e['dice'] for e in visible]),
        visibility_accuracy=mean([e['pred_visible'] is not None and e['pred_visible']==e['gt_visible'] for e in entries]),
        visibility_abstentions=sum(e['pred_visible'] is None for e in entries),
        invisible_false_positive_rate=mean([e['pred_visible'] is True for e in invisible]),
        out_of_view_samples=len(out),out_of_view_residue_rate=mean([e['pred_visible'] is True for e in out]),
        out_of_view_residue_pixels=sum(e['pred_area'] for e in out))


def temporal_metrics(sequence,stride=5,reentry_window=3):
    """One manually labeled clip/object. No interpolation across unaudited frames."""
    runs=[];run=[];events=[]
    def close():
        if run:
            runs.append(dict(samples=len(run),span_seconds=max(0,run[-1]['timestamp']-run[0]['timestamp']),
                first_frame=run[0]['frame_idx'],last_frame=run[-1]['frame_idx'],duration_is_lower_bound=True))
            run.clear()
    for i,e in enumerate(sequence):
        if i and e['frame_idx']-sequence[i-1]['frame_idx']!=stride:close()
        if e['gt_visibility'] in INVISIBLE and e['pred_visible'] is True:run.append(e)
        else:close()
        if i and e['gt_visible'] and sequence[i-1]['gt_visibility']=='out_of_view' and e['frame_idx']-sequence[i-1]['frame_idx']==stride:
            window=sequence[i:i+reentry_window]
            contiguous=all(b['frame_idx']-a['frame_idx']==stride for a,b in zip(window,window[1:]))
            known=all(x['gt_visibility'] in VISIBLE for x in window)
            successes=[x for x in window if x['gt_visible'] and x['iou'] is not None and x['iou']>=.5]
            # Right-censored or re-occluded events are not counted as failed recovery.
            eligible=contiguous and known and (len(window)==reentry_window or bool(successes))
            events.append(dict(frame_idx=e['frame_idx'],eligible=eligible,success=bool(successes) if eligible else None,
                latency_seconds=successes[0]['timestamp']-e['timestamp'] if eligible and successes else None))
    close()
    return runs,events


def match_identities(predictions,labels):
    gt=[g for g in labels if g['visibility'] in VISIBLE]
    pred=[r for r in predictions if r['visible'] is True and r['mask'] is not None]
    if not gt or not pred:return []
    masks=[decode(r['mask']) for r in pred];truth=[decode(g['mask']) for g in gt]
    matrix=np.array([[similarity(a,b)[0] or 0. for b in truth] for a in masks])
    pi,gi=linear_sum_assignment(-matrix)
    return [dict(instance_id=gt[j]['instance_id'],gt_object_id=gt[j]['object_id'],
        pred_track_id=pred[i]['track_id'],pred_object_id=pred[i]['object_id'],iou=float(matrix[i,j]))
        for i,j in zip(pi,gi) if matrix[i,j]>=.1]


def evaluate(manifest,labels,predictions):
    results={};details=[]
    lookup={(g['sample_id'],g['object_id']):g for g in labels}
    for mode,bykey in predictions.items():
        entries=[];seq=defaultdict(list);ids_by_clip=defaultdict(list);coverage=Counter()
        for s in manifest['samples']:
            labels_here=[]
            for o in s['objects']:
                key=(s['sample_id'],o['object_id']);g=lookup.get(key)
                if g is None:continue
                labels_here.append(g)
                pk=(s['episode_id'],s['camera'],s['frame_idx'],o['object_id'])
                assert pk in bykey, f'Missing prediction (not an absent-object row): {mode} {pk}'
                value=pair_score(bykey[pk],g)|dict(camera=s['camera'],sample_id=s['sample_id'],frame_idx=s['frame_idx'],
                    timestamp=s['timestamp'],clip_id=s['clip_id'],object_id=o['object_id'],conditions=g.get('conditions',[]))
                if g['visibility']!='unknown':entries.append(value)
                seq[(s['clip_id'],o['object_id'])].append(value)
                coverage.update(g.get('conditions',[]))
            # Full scene labels required for class-agnostic identity assignment.
            if len(labels_here)==len(s['objects']) and all(g['visibility']!='unknown' for g in labels_here):
                assert len({g['instance_id'] for g in labels_here})==len(labels_here),'Instance IDs must be distinct within a frame'
                rs=[bykey[(s['episode_id'],s['camera'],s['frame_idx'],o['object_id'])] for o in s['objects']]
                ids_by_clip[s['clip_id']].append(s|dict(matches=match_identities(rs,labels_here)))
        bycam={}
        for camera in ['head','left_wrist','right_wrist']:
            es=[e for e in entries if e['camera']==camera];metric=aggregate(es)
            fp=[];reentries=[]
            for (clip,oid),values in seq.items():
                values=sorted(values,key=lambda x:x['frame_idx'])
                if values[0]['camera']!=camera:continue
                rr,ee=temporal_metrics(values);fp.extend(rr);reentries.extend(ee)
                details.append(dict(mode=mode,clip_id=clip,object_id=oid,false_positive_runs=rr,reentry_events=ee))
            eligible=[e for e in reentries if e['eligible']]
            comparisons=0;switches=0;matches=0;correct=0;id_events=[]
            for clip,values in ids_by_clip.items():
                if values[0]['camera']!=camera:continue
                previous={};previous_frame=None
                for s in sorted(values,key=lambda x:x['frame_idx']):
                    if previous_frame is not None and s['frame_idx']-previous_frame!=5:previous={}
                    for match in s['matches']:
                        identity=match['instance_id'];track=match['pred_track_id'];matches+=1
                        correct+=int(match['pred_object_id']==match['gt_object_id'])
                        if identity in previous:
                            comparisons+=1
                            if previous[identity]!=track:
                                switches+=1;id_events.append(dict(sample_id=s['sample_id'],instance_id=identity,before=previous[identity],after=track))
                        previous[identity]=track
                    previous_frame=s['frame_idx']
            metric.update(id_switch_count=switches if matches else None,id_comparisons=comparisons,
                id_switch_rate=switches/comparisons if comparisons else None,
                matched_semantic_identity_accuracy=correct/matches if matches else None,
                matched_instances=matches,false_positive_runs=len(fp),
                false_positive_max_consecutive_samples=max((r['samples'] for r in fp),default=None),
                false_positive_max_span_seconds=max((r['span_seconds'] for r in fp),default=None),
                reentry_events=len(reentries),eligible_reentry_events=len(eligible),
                reentry_success_rate=sum(e['success'] for e in eligible)/len(eligible) if eligible else None,
                reentry_latency_seconds_mean=float(np.mean([e['latency_seconds'] for e in eligible if e['success']])) if any(e['success'] for e in eligible) else None,
                id_switch_events=id_events)
            bycam[camera]=metric
        bycondition={c:aggregate([e for e in entries if c in e['conditions']]) for c in manifest['required_conditions']}
        results[mode]=dict(overall=aggregate(entries),per_camera=bycam,per_condition=bycondition,condition_labeled_object_frames=dict(coverage))
    required=sum(len(s['objects']) for s in manifest['samples'])
    status='PENDING_HUMAN_GT' if not labels else 'PARTIAL_HUMAN_AUDIT' if len(labels)<required else 'HUMAN_AUDIT_COMPLETE'
    return dict(status=status,human_labels=len(labels),required_labels=required,frames=len(manifest['samples']),results=results,
        interpretation='null means unmeasured/undefined, not zero error. Challenge clips are not a random population sample.',
        definitions=dict(iou_dice='macro over visible/partly occluded object-frames; misses score zero; invisible GT excluded',
            visibility='unknown GT excluded; unknown predictions count incorrect and as abstentions',
            identity='class-agnostic Hungarian mask IoU>=0.1 matching on fully labeled frames; compare GT instance -> stable semantic track ID within each audited clip; re-seed segment is not a new ID',
            false_positive='visible prediction on fully occluded/out-of-view GT; persistence is sampled audited span, not full-video lifetime',
            reentry='out_of_view -> visible within a clip; intended-object IoU>=0.5 within 3 sampled frames; right-censored/re-occluded windows excluded')),details


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.audit/'manifest.json').read_text());samples={s['sample_id']:s for s in manifest['samples']}
    labels=[]
    for path in sorted((a.audit/'human_labels').glob('*.json')):
        g=json.loads(path.read_text());s=samples[g['sample_id']]
        assert g['human_confirmed'] is True and g['annotation_method']=='manual_pixel_editor'
        assert g['source_image_sha256']==s['image_sha256']==sha(a.audit/s['image'])
        mask=decode(g['mask']);assert mask.shape==(s['image_height'],s['image_width'])
        assert bool(mask.any())==(g['visibility'] in VISIBLE)
        assert g['object_id'] in [o['object_id'] for o in s['objects']]
        labels.append(g)
    assert len({(g['sample_id'],g['object_id']) for g in labels})==len(labels)
    predictions={}
    for mode in ['baseline','recovery']:
        lookup={}
        for path in a.run.glob(f'task_*/episode_*/*.{mode}.jsonl'):
            for row in map(json.loads,path.read_text().splitlines()):
                key=(row['episode_id'],row['camera'],row['frame_idx'],row['object_id'])
                assert key not in lookup;lookup[key]=row
                if row['mask'] is not None:assert bbox(decode(row['mask']))==row['bbox_xyxy']
        assert lookup, f'No {mode} outputs'
        predictions[mode]=lookup
    summary,details=evaluate(manifest,labels,predictions)
    summary['provenance']=dict(manifest_sha256=sha(a.audit/'manifest.json'),run_config_sha256=sha(a.run/'run_config.json'),
        script_sha256=sha(__file__),label_hashes={p.name:sha(p) for p in (a.audit/'human_labels').glob('*.json')})
    write_json(a.output/'metrics.json',summary);write_json(a.output/'temporal_details.json',details)
    fields=['mode','camera','labeled_object_frames','mask_iou','mask_dice','visibility_accuracy','id_switch_count','id_switch_rate','reentry_success_rate']
    with (a.output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for mode,result in summary['results'].items():
            for camera,values in result['per_camera'].items():writer.writerow(dict(mode=mode,camera=camera,**{k:values[k] for k in fields[2:]}))
    lines=['# Human mask audit metrics',f'Status: {summary["status"]}. Human labels: {len(labels)}/{summary["required_labels"]}.',
           'Null/blank scores are unmeasured, not zero error. See metrics.json for definitions, denominators and condition groups.',
           '| Mode | Camera | IoU | Dice | Visibility accuracy | ID switches | Re-entry success |',
           '|---|---|---|---|---|---|---|']
    for mode,result in summary['results'].items():
        for camera,v in result['per_camera'].items():
            fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
            lines.append('| '+' | '.join([mode,camera]+[fmt(v[k]) for k in ['mask_iou','mask_dice','visibility_accuracy','id_switch_count','reentry_success_rate']])+' |')
    (a.output/'report.md').write_text('\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n')
    print(json.dumps(dict(status=summary['status'],human_labels=len(labels),required=summary['required_labels'])))


if __name__=='__main__':main()
