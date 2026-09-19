"""SAM image proposals from explicit pixel prompts; inspect before propagation."""
import argparse
import json
import os
from pathlib import Path
import sys

from common import load_clip, save, sha

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'models/sam2/sam2.1_hiera_base_plus.pt')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    path, clip = load_clip(a.clip)
    prompts = json.loads(a.prompts.read_text())
    assert prompts['clip_id'] == clip['clip_id'] and prompts['coordinate_system'] == 'source_image_pixels'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible and visible != '-1' and ',' not in visible, 'Select one checked available GPU'
    dest = a.output.resolve()
    assert not dest.is_relative_to(path.parent) and not dest.is_relative_to(ROOT/'examples')
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sys.path.insert(0, str(ROOT/'tools/object_annotation'))
    from masks import encode
    torch.set_num_threads(2)
    dest.mkdir(parents=True, exist_ok=False)
    model = build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml', str(a.checkpoint), device='cuda', apply_postprocessing=False)
    model.eval().requires_grad_(False)
    predictor = SAM2ImagePredictor(model)
    seeds = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for j, spec in enumerate(prompts['seeds']):
            frame = clip['frames'][spec['local_frame_idx']]
            assert frame['image_sha256'] == spec['source_image_sha256']
            image = np.asarray(Image.open(path.parent/frame['image']))
            h, w = image.shape[:2]
            pts = np.asarray(spec['points'], np.float32)
            labels = np.asarray(spec['point_labels'], np.int32)
            assert pts.ndim == 2 and pts.shape[1] == 2 and len(pts) == len(labels)
            assert np.isfinite(pts).all() and ((pts >= 0) & (pts < [w, h])).all()
            assert set(labels.tolist()) <= {0, 1} and 1 in labels
            box = np.asarray(spec['box_xyxy'], np.float32) if spec.get('box_xyxy') else None
            if box is not None:
                assert box.shape == (4,) and np.isfinite(box).all()
                assert 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=labels, box=box, multimask_output=False)
            binary = masks[0].astype(bool)
            assert binary.any(), 'Empty seed proposal; review RGB before choosing absence'
            overlay = image.copy(); overlay[binary] = (.5*overlay[binary]+np.array([125, 110, 10])).clip(0, 255).astype('uint8')
            Image.fromarray(np.concatenate([image, overlay], axis=1)).save(dest/f'seed_{j:03d}.png')
            seeds.append(dict(local_frame_idx=spec['local_frame_idx'], source_frame_idx=frame['source_frame_idx'],
                object_id=spec['object_id'], mask=encode(binary), source_image_sha256=frame['image_sha256'],
                human_confirmed=False, prompt=spec, sam_predicted_iou=float(scores[0])))
    save(dest/'seeds.proposed.json', dict(schema='mask_handoff.seeds.v1', clip_id=clip['clip_id'],
        visual_review=False, review_status='view_seed_overlays_before_creating_reviewed_copy', seeds=seeds,
        prompts_sha256=sha(a.prompts), checkpoint_sha256=sha(a.checkpoint)))
    print(json.dumps(dict(output=str(dest), proposed_seeds=len(seeds), propagated=False), indent=2))


if __name__ == '__main__':
    main()
