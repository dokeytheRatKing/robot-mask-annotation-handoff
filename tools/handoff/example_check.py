"""Validate real example masks, render QA and reconstruct released ROI semantics."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from common import load_clip, load_mask_seeds, save, sha

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/object_annotation'))
from masks import bbox, decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    dest = a.output.resolve()
    if dest.is_relative_to(ROOT/'examples'):
        raise ValueError('Output must not modify source examples')
    dest.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    cards = []
    row_count = 0
    unknown_count = 0
    for name in ['kettle_wrist', 'fruit_reentry']:
        path, clip = load_clip(ROOT/'examples'/name/'clip.json')
        load_mask_seeds(path.parent/'seeds.json', clip)
        layers = {key: [json.loads(line) for line in (path.parent/f'{key}.jsonl').read_text().splitlines()]
                  for key in ['objects', 'robot_parts']}
        metadata = {r['frame_idx']: r for r in json.loads((path.parent/'frames.json').read_text())}
        rois, knowns = [], []
        for frame in clip['frames']:
            image = np.array(Image.open(path.parent/frame['image']))
            h, w = image.shape[:2]
            overlay = image.copy()
            foreground = np.zeros((h, w), bool)
            for layer, rows in layers.items():
                selected = [r for r in rows if r['frame_idx'] == frame['source_frame_idx']]
                for row in selected:
                    row_count += 1
                    assert row['timestamp'] == frame['timestamp']
                    assert row['camera'] == clip['camera'] and row['episode_id'] == clip['episode_id']
                    if row['mask'] is None:
                        unknown_count += 1
                        assert row['visible'] is None and not row['training_eligible']
                        continue
                    binary = decode(row['mask'])
                    assert binary.shape == (h, w)
                    assert int(binary.sum()) == row['mask_area'] and bbox(binary) == row['bbox_xyxy']
                    color = np.random.default_rng(row['object_id']+7).integers(60, 245, 3)
                    overlay[binary] = (.5*overlay[binary]+.5*color).astype('uint8')
                    if row['training_eligible']:
                        assert row['object_id'] < 1000 or row['object_id'] in (1103, 1104)
                        foreground |= binary
            meta = metadata[frame['source_frame_idx']]
            radius = meta['margin_radius_pixels']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            roi = cv2.dilate(foreground.astype('uint8'), kernel).astype(bool)
            known = np.ones_like(roi) if not meta['unresolved_object_ids'] else roi.copy()
            assert roi.sum() == meta['roi_pixels'] and known.sum() == meta['known_pixels']
            rois.append(roi); knowns.append(known)
            roi_rgb = np.repeat(roi[..., None].astype('uint8')*255, 3, axis=2)
            known_rgb = np.repeat(known[..., None].astype('uint8')*255, 3, axis=2)
            canvas = np.concatenate([np.concatenate([image, overlay], 1), np.concatenate([roi_rgb, known_rgb], 1)], 0)
            file = f'{name}_{frame["source_frame_idx"]}.jpg'
            Image.fromarray(canvas).save(dest/file, quality=90)
            cards.append(f'<h2>{name} source frame {frame["source_frame_idx"]}</h2><p>RGB | objects + independent robot; ROI | known</p><img src="{file}">')
        np.savez_compressed(dest/f'{name}_task_roi.npz', roi=np.stack(rois), known=np.stack(knowns),
            source_frame_idx=np.array([r['source_frame_idx'] for r in clip['frames']]))
    audit = ROOT/'examples/confirmed_audit'
    samples = json.loads((audit/'manifest.json').read_text())['samples']
    label_count = 0
    for sample in samples:
        assert sha(audit/sample['image']) == sample['image_sha256']
        image = np.array(Image.open(audit/sample['image']))
        overlay = image.copy()
        for obj in sample['objects']:
            label = json.loads((audit/'human_labels'/f'{sample["sample_id"]}__{obj["object_id"]}.json').read_text())
            assert label['human_confirmed'] and label['source_image_sha256'] == sample['image_sha256']
            binary = decode(label['mask'])
            assert binary.shape == image.shape[:2]
            color = np.random.default_rng(obj['object_id']+7).integers(60, 245, 3)
            overlay[binary] = (.5*overlay[binary]+.5*color).astype('uint8')
            label_count += 1
        file = sample['sample_id']+'.jpg'
        Image.fromarray(np.concatenate([image, overlay], 1)).save(dest/file, quality=90)
        cards.append(f'<h2>Confirmed: {sample["sample_id"]}</h2><img src="{file}">')
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Real mask examples</title><style>body{max-width:1300px;margin:auto;font:16px sans-serif}img{width:100%}</style>'+''.join(cards))
    report = dict(status='PASS', clip_frames=16, released_rows=row_count, null_unknown_rows=unknown_count,
        exact_confirmed_example_labels=label_count, roi_reconstruction=True, gpu_used=False,
        meaning='Engineering example validation; no new semantic or human acceptance')
    save(dest/'validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
