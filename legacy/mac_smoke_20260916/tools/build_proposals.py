"""Assemble visually audited SAM masks as unconfirmed, reproducible proposals.

Human labels are read-only inputs. Scene decisions are explicit in
work/segmentation/scene_decisions.json, rather than inferred from empty masks.
"""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from assist_masks import AUDIT, WORK, encode, decode, human_rows


def contains(index, intervals):
    return any(a <= index <= b for a, b in intervals)


def main():
    manifest = json.loads((AUDIT / 'manifest.json').read_text())
    humans = human_rows()
    decisions_path = WORK / 'scene_decisions.json'
    decisions = json.loads(decisions_path.read_text())
    predictions = {}
    for path in (WORK / 'predictions').glob('*.npz'):
        z = np.load(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        packed = z['masks']
        for j, i in enumerate(z['ui_indices']):
            for k, oid in enumerate(z['object_ids']):
                predictions[int(i), int(oid)] = (packed[j, k], meta)
        z.close()
    dest = AUDIT / 'proposed_labels'
    dest.mkdir(exist_ok=True)
    summaries, changes = [], []
    created = datetime.now(timezone.utc).isoformat()
    for i, sample in enumerate(manifest['samples'], 1):
        shape = (sample['image_height'], sample['image_width'])
        entries = {}
        for obj in sample['objects']:
            oid = obj['object_id']
            key = sample['sample_id'], oid
            if key in humans:
                entries[oid] = dict(row=humans[key], mask=decode(humans[key]['mask']), priority=3)
                continue
            scene = decisions['objects'][str(oid)]
            visible = contains(i, scene['present'])
            status = ('partial_occlusion' if contains(i, scene['partial']) else 'visible') if visible else 'out_of_view'
            if contains(i, scene.get('fully_occluded', [])):
                assert not visible
                status = 'fully_occluded'
            if contains(i, scene.get('unknown', [])):
                assert not visible
                status = 'unknown'
            mask, meta = predictions.get((i, oid), (np.zeros(shape, bool), {}))
            mask = mask.copy()
            refinement = WORK / 'refinements' / f'{i:03d}_{oid}.npy'
            method = 'sam2_1_video_assisted'
            priority = 0
            if refinement.exists():
                mask = np.load(refinement)
                method = 'sam2_1_image_refined'
                priority = 2
            if not visible:
                if mask.any():
                    changes.append(dict(ui_index=i, object_id=oid, action='remove_false_positive', area=int(mask.sum())))
                mask[:] = False
                method = 'visual_sequence_review_empty'
            assert bool(mask.any()) == visible, f'Missing visible mask: {i}, {oid}'
            conditions = []
            if status == 'partial_occlusion':
                conditions.append('partial_occlusion')
            elif status == 'visible':
                conditions.append('normal_unoccluded')
            elif status == 'out_of_view':
                conditions.append('out_of_view')
            for tag in ['grasp_contact', 'reentry', 'similar_objects_near', 'wrist_closeup']:
                if contains(i, scene.get(tag, [])):
                    assert visible, (i, oid, tag)
                    conditions.append(tag)
            notes = '自动辅助草稿，待人工核对可见表面边界、visibility 和场景标签。'
            if status == 'unknown':
                notes += '物体在夹爪与画面下边缘附近消失，完全遮挡或完全出画尚待确认。'
            if (i >= 281 and i <= 284) or i in [164, 193, 221, 416]:
                notes += '运动模糊或细小边缘，建议放大检查。'
            seed_rows = meta.get('seeds', [])
            provenance = dict(model='SAM 2.1 Hiera Small', checkpoint_sha256=meta.get('checkpoint_sha256'),
                seed=20260916, device='mps', clip_id=sample['clip_id'],
                human_seed_count=sum(r['source'] == 'human' for r in seed_rows),
                visual_prompt_count=sum(r['source'] == 'visual_prompt' for r in seed_rows),
                prediction_metadata=f'work/segmentation/predictions/{sample["clip_id"]}.json' if meta else None,
                refinement_prompt=f'work/segmentation/refinements/{i:03d}_{oid}.json' if refinement.exists() else None,
                scene_decisions_sha256=hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
                annotation_method=method)
            proposal_id = f'sam21-20260916-{sample["sample_id"]}-{oid}'
            provenance['proposal_id'] = proposal_id
            row = dict(sample_id=sample['sample_id'], episode_id=sample['episode_id'], camera=sample['camera'],
                frame_idx=sample['frame_idx'], object_id=oid, instance_id=obj['class_name'].replace(' ', '_')+'_1',
                visibility=status, conditions=conditions, annotator='', human_confirmed=False,
                prediction_prefill=True, annotation_method=method, source_image_sha256=sample['image_sha256'],
                notes=notes, generated_utc=created, proposal_id=proposal_id, model='sam2.1_hiera_small',
                review_status='pending_human_review', proposal_provenance=provenance)
            entries[oid] = dict(row=row, mask=mask, priority=priority)
        # Human masks take priority; refined fruit pixels also take priority over
        # the video's basket masks. Never modify existing human annotations.
        occupied = np.zeros(shape, bool)
        order = sorted(entries, key=lambda oid: (entries[oid]['priority'], oid != 4), reverse=True)
        for oid in order:
            e = entries[oid]
            if not e['row']['human_confirmed']:
                removed = int((e['mask'] & occupied).sum())
                e['mask'] &= ~occupied
                if removed:
                    changes.append(dict(ui_index=i, object_id=oid, action='remove_overlap', area=removed))
            occupied |= e['mask']
        for oid, e in entries.items():
            row, mask = e['row'], e['mask']
            assert bool(mask.any()) == (row['visibility'] in ['visible', 'partial_occlusion']), (i, oid)
            if not row['human_confirmed']:
                row['mask'] = encode(mask)
                row['mask_area'] = int(mask.sum())
                path = dest / f'{sample["sample_id"]}__{oid}.json'
                tmp = path.with_suffix('.json.tmp')
                tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
                tmp.replace(path)
            summaries.append(dict(ui_index=i, sample_id=sample['sample_id'], object_id=oid,
                source='human' if row['human_confirmed'] else 'proposal', visibility=row['visibility'],
                mask_area=int(mask.sum()), conditions=';'.join(row['conditions']), notes=row['notes']))
    with (WORK / 'label_inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    report = dict(total=len(summaries), sources=dict(Counter(r['source'] for r in summaries)),
        visibility=dict(Counter(r['visibility'] for r in summaries)),
        uncertain=[r for r in summaries if r['visibility']=='unknown'], corrections=changes)
    (WORK / 'proposal_build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'corrections'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
