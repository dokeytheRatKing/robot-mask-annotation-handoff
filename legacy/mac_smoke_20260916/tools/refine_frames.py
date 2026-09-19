"""SAM 2.1 image refinement for visually checked tracking failures.

Prompt coordinates use a 640 x 360 reference canvas. Crops improve the
resolution of very small visible fragments. Results remain proposals.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from assist_masks import ROOT, AUDIT, WORK


def main():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.manual_seed(20260916)
    torch.set_num_threads(4)
    predictor = SAM2ImagePredictor(build_sam2(
        'configs/sam2.1/sam2.1_hiera_s.yaml',
        str(ROOT / 'models/sam2.1_hiera_small.hf.pt'), device='mps',
        apply_postprocessing=False))
    samples = json.loads((AUDIT / 'manifest.json').read_text())['samples']
    prompts = json.loads((WORK / 'refinement_prompts.json').read_text())
    dest = WORK / 'refinements'
    dest.mkdir(exist_ok=True)
    for p in prompts:
        s = samples[p['ui_index'] - 1]
        arr = np.array(Image.open(AUDIT / s['image']).convert('RGB'))
        scale = np.array([arr.shape[1]/640, arr.shape[0]/360])
        crop = (np.array(p.get('crop', [0, 0, 640, 360])) * np.tile(scale, 2)).astype(int)
        x0, y0, x1, y1 = crop
        origin = np.array([x0, y0])
        points = np.array(p['points']) * scale - origin
        box = np.array(p['box']) * np.tile(scale, 2) - np.tile(origin, 2)
        with torch.inference_mode():
            predictor.set_image(arr[y0:y1, x0:x1])
            masks, scores, _ = predictor.predict(
                point_coords=points, point_labels=np.array(p['labels']),
                box=box, multimask_output=False)
        mask = np.zeros(arr.shape[:2], bool)
        mask[y0:y1, x0:x1] = masks[0]
        np.save(dest / f'{p["ui_index"]:03d}_{p["object_id"]}.npy', mask)
        (dest / f'{p["ui_index"]:03d}_{p["object_id"]}.json').write_text(json.dumps(
            dict(prompt=p, area=int(mask.sum()), model_score=float(scores[0])), indent=2))
        print(p['ui_index'], p['object_id'], int(mask.sum()), float(scores[0]), flush=True)


if __name__ == '__main__':
    main()
