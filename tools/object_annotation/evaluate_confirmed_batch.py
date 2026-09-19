"""Compare frozen review snapshots with confirmed masks, without temporal claims."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from annotate import sha, write_json
from masks import decode

PART_IDS = (1101, 1102, 1103, 1104)


def prediction_mask(label, predictions):
    shape = (label['image_height'], label['image_width'])
    result = np.zeros(shape, bool)
    group = label['group']
    if group == 'objects':
        ids = label.get('merged_object_ids') or [label['object_id']]
        lookup = {r['object_id']: r for r in predictions['objects']}
    else:
        ids = PART_IDS if group == 'robot' else [label['object_id']]
        lookup = {r['part_id']: r for r in predictions['robot_parts']}
    for oid in ids:
        row = lookup[oid]  # Missing predictions are a schema failure, not an empty mask.
        binary = decode(row['mask'])
        assert binary.shape == shape
        result |= binary
    return result


def score(pred, gt):
    intersection = int((pred & gt).sum())
    pa, ga = int(pred.sum()), int(gt.sum())
    union = pa + ga - intersection
    return dict(pred_pixels=pa, gt_pixels=ga, intersection=intersection, union=union,
                iou=intersection / union if union else None,
                dice=2 * intersection / (pa + ga) if pa + ga else None,
                visibility_correct=bool(pa) == bool(ga),
                false_positive=bool(pa) and not ga, missed=bool(ga) and not pa)


def aggregate(rows):
    visible = [r for r in rows if r['gt_pixels']]
    invisible = [r for r in rows if not r['gt_pixels']]
    union = sum(r['union'] for r in rows)
    return dict(labels=len(rows), gt_visible=len(visible), gt_invisible=len(invisible),
                mean_iou_visible=float(np.mean([r['iou'] for r in visible])) if visible else None,
                mean_dice_visible=float(np.mean([r['dice'] for r in visible])) if visible else None,
                micro_iou=sum(r['intersection'] for r in rows) / union if union else None,
                visibility_accuracy=sum(r['visibility_correct'] for r in rows) / len(rows) if rows else None,
                false_positives=sum(r['false_positive'] for r in rows),
                misses=sum(r['missed'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accepted', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    labels_path = a.accepted / 'accepted_masks.jsonl'
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    cases, rows, excluded = {}, [], []
    for label in labels:
        sid = label['sample_id']
        if sid not in cases:
            path = a.snapshot / sid / 'case.json'
            cases[sid] = json.loads(path.read_text())
            assert sha(a.snapshot / sid / 'rgb.png') == label['provenance']['source_image_sha256']
        case = cases[sid]
        assert label['human_confirmed']
        assert all(label[k] == case[k] for k in ('episode_id', 'camera', 'frame_idx', 'task_id'))
        if label['exclude_from_metrics']:
            excluded.append(dict(sample_id=sid, object_id=label['object_id'], reason=label['visibility']))
            continue
        gt = decode(label['mask'])
        assert bool(gt.any()) == label['visible']
        rows.append(dict(sample_id=sid, task_id=label['task_id'], camera=label['camera'],
                         group=label['group'], object_id=label['object_id'], class_name=label['class_name'],
                         **score(prediction_mask(label, case['predictions']), gt)))
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['camera']}/{row['group']}"].append(row)
    metrics = dict(frames=len(cases), evaluated_labels=len(rows), excluded=excluded,
                   by_camera_layer={key: aggregate(value) for key, value in sorted(groups.items())},
                   by_layer={key: aggregate([r for r in rows if r['group'] == key])
                             for key in ('objects', 'robot_parts', 'robot')},
                   temporal_metrics=None,
                   limitations=['Biased hard-case set; not corpus accuracy.',
                                'One frame per stream: ID switches, persistence and recovery unmeasured.',
                                'Visible-label macro IoU/Dice; empty GT labels enter visibility and micro IoU.',
                                'Old unknown robot part excluded; robot union is four named parts only.'],
                   provenance=dict(accepted_sha256=sha(labels_path), script_sha256=sha(__file__),
                                   snapshots={sid: sha(a.snapshot/sid/'case.json') for sid in cases}))
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output/'metrics.json', metrics)
    with (a.output/'per_label.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = ['# Frozen prediction hard-case diagnostic', '',
            f'{len(cases)} frames; {len(rows)} evaluated labels; {len(excluded)} ignored labels.', '',
            '| Camera/layer | Labels | Visible GT | IoU | Dice | Visibility accuracy | FP | Miss |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, v in metrics['by_camera_layer'].items():
        text.append(f"| {key} | {v['labels']} | {v['gt_visible']} | {v['mean_iou_visible']:.4f} | "
                    f"{v['mean_dice_visible']:.4f} | {v['visibility_accuracy']:.4f} | "
                    f"{v['false_positives']} | {v['misses']} |")
    text += ['', *metrics['limitations']]
    (a.output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'provenance'}, indent=2))


if __name__ == '__main__':
    main()
