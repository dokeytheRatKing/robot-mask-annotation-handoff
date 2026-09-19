"""Freeze 420 RGB frames in contiguous audit clips; NO generated GT labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
from annotate import BASE,PROJECT,write_json,sha
from data import frames


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output;root.mkdir(parents=True,exist_ok=False)
    source=PROJECT/'annotations/groundingdino_pilot_20260916/selected_episodes.json'
    episodes={e['episode_id']:e for e in json.loads(source.read_text())['episodes']}
    objects={o['object_id']:o for o in json.loads((BASE/'config/objects.json').read_text())}
    mapping=json.loads((BASE/'config/task_objects.json').read_text())
    # Selection tags are hypotheses, NOT human-verified condition labels.
    main_starts={'head':[0,300,450,850,1000],'left_wrist':[0,180,420,650,835],
                 'right_wrist':[0,180,470,900,1095]}
    selection={'episode_002478':main_starts,
               'episode_001608':{c:[100] for c in main_starts},
               'episode_002577':{c:[250] for c in main_starts}}
    planned=['normal_unoccluded','grasp_contact','partial_occlusion','out_of_view','reentry','similar_objects_near','wrist_closeup']
    samples=[];clips=[];source_hashes={}
    for epid,bycam in selection.items():
        ep=episodes[epid]
        for camera,starts in bycam.items():
            selected={}
            for start in starts:
                clip=f'{epid}_{camera}_{start:06d}'
                ids=list(range(start,start+100,5));assert ids[-1]<ep['frames']
                clips.append(dict(clip_id=clip,episode_id=epid,camera=camera,frame_indices=ids,
                    selection_reason='fixed failure-context or coverage clip; condition tags require human confirmation'))
                for idx in ids:assert idx not in selected;selected[idx]=clip
            source_hashes[ep['cameras'][camera]['video']]=sha(ep['cameras'][camera]['video'])
            for idx,ts,image in frames(ep,camera,5):
                if idx not in selected:continue
                sample_id=f'{epid}_{camera}_{idx:06d}';rel=Path('images')/(sample_id+'.png')
                (root/rel.parent).mkdir(exist_ok=True)
                assert cv2.imwrite(str(root/rel),image)
                samples.append(dict(sample_id=sample_id,clip_id=selected[idx],episode_id=epid,task_id=ep['task_id'],
                    camera=camera,frame_idx=idx,timestamp=ts,image=str(rel),image_sha256=sha(root/rel),
                    image_width=image.shape[1],image_height=image.shape[0],
                    objects=[dict(object_id=oid,class_name=objects[oid]['class_name']) for oid in mapping[str(ep['task_id'])]],
                    annotation_status='PENDING_HUMAN',split='diagnostic_audit_not_independent_test'))
    assert len(samples)==420 and Counter(s['camera'] for s in samples)==dict.fromkeys(main_starts,140)
    write_json(root/'manifest.json',dict(schema='astribot.human_mask_audit.v1',samples=samples,clips=clips,
        selection_seed=None,selection_policy='fixed clips chosen before recovery results; deliberately difficult, not population-random',
        required_conditions=planned,condition_coverage='UNCONFIRMED_UNTIL_HUMAN_REVIEW',
        baseline_reference='annotations/grounded_sam2_mask_pilot_20260916',
        source_manifest_sha256=sha(source),source_videos_sha256=source_hashes,script_sha256=sha(__file__)))
    write_json(root/'status.json',dict(frames=420,frames_per_camera=140,episodes=len(selection),
        object_frame_labels_required=sum(len(s['objects']) for s in samples),human_labels=0,
        status='PENDING_HUMAN_ANNOTATION',true_quality_metrics=None))
    print(root/'manifest.json')


if __name__=='__main__':main()
