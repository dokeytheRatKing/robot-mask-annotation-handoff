"""Reproducible SAM 2.1 proposals, kept separate from human annotations.

Run with the astribot-sam21 conda environment. Inputs remain original PNGs;
numbered .jpg symlinks only satisfy SAM's filename filter (Pillow decodes PNG).
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'astribot_mask_audit_420_20260916'
AUDIT = BUNDLE / 'audit'
WORK = ROOT / 'work/segmentation'
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(ROOT / 'tools/sam2'))
from audit_rle import decode_counts, encode_counts


def decode(rle):
    counts = decode_counts(rle)
    pixels = np.zeros(np.prod(rle['size']), dtype=np.uint8)
    offset = 0
    for i, count in enumerate(counts):
        if i % 2:
            pixels[offset:offset + count] = 1
        offset += count
    return pixels.reshape(rle['size'], order='F').astype(bool)


def encode(mask):
    flat = mask.astype(np.uint8).ravel(order='F')
    breaks = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, breaks, len(flat)]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return encode_counts(counts, *mask.shape)


def human_rows():
    return {(r['sample_id'], r['object_id']): r
            for p in (AUDIT / 'human_labels').glob('*.json')
            for r in [json.loads(p.read_text())]}


def main():
    from sam2.build_sam import build_sam2_video_predictor
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips', nargs='+', type=int)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    torch.set_num_threads(4)
    manifest = json.loads((AUDIT / 'manifest.json').read_text())
    rows = human_rows()
    extra_path = WORK / 'extra_seeds.json'
    extra = json.loads(extra_path.read_text()) if extra_path.exists() else []
    samples = {s['sample_id']: (i + 1, s) for i, s in enumerate(manifest['samples'])}
    ckpt = ROOT / 'models/sam2.1_hiera_small.hf.pt'
    model = build_sam2_video_predictor(
        'configs/sam2.1/sam2.1_hiera_s.yaml', str(ckpt), device=args.device,
        apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0', '++model.non_overlap_masks=true'],
    )
    if args.device == 'mps':
        # SAM stores video memory in bfloat16. When an object's only seed is in
        # a future frame, no float32 object pointer is concatenated to promote
        # that memory; MPS then aborts inside mixed-dtype attention matmul.
        # Normalize attention inputs to the model's float32 inference dtype.
        def float_memory(module, positional, keyword):
            if 'memory' in keyword:
                keyword['memory'] = keyword['memory'].float()
            return positional, keyword
        model.memory_attention.register_forward_pre_hook(float_memory, with_kwargs=True)
    print('MODEL_READY', args.device, flush=True)
    for clip_no, clip in enumerate(manifest['clips'], 1):
        if args.clips and clip_no not in args.clips:
            continue
        clip_id = clip['clip_id']
        dest = WORK / 'predictions' / (clip_id + '.npz')
        if dest.exists() and not args.force:
            print('SKIP', clip_id, flush=True)
            continue
        clip_samples = [(idx, s) for idx, s in samples.values() if s['clip_id'] == clip_id]
        lookup = {s['sample_id']: j for j, (_, s) in enumerate(clip_samples)}
        local_extra = [r for r in extra if r['ui_index'] in [idx for idx, _ in clip_samples]]
        seeds = [(lookup[sid], oid, decode(r['mask']), {'source': 'human', 'sample_id': sid})
                 for (sid, oid), r in rows.items() if sid in lookup]
        # Empty masks alone do not identify a target. Only track objects having a
        # positive human mask or explicit visual prompt somewhere in this clip.
        positive = {oid for _, oid, mask, _ in seeds if mask.any()}
        positive.update(r['object_id'] for r in local_extra if not r.get('empty'))
        seeds = [seed for seed in seeds if seed[1] in positive]
        if not positive:
            print('NO_POSITIVE_SEEDS', clip_no, clip_id, flush=True)
            continue
        frame_dir = WORK / 'frames' / clip_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        for j, (_, s) in enumerate(clip_samples):
            link = frame_dir / f'{j:05d}.jpg'
            if not link.exists():
                link.symlink_to((AUDIT / s['image']).resolve())
        started = time.time()
        print('START', clip_no, clip_id, 'objects', sorted(positive), flush=True)
        with torch.inference_mode():
            state = model.init_state(str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
            seed_meta = []
            for j, oid, mask, meta in sorted(seeds, key=lambda x: (x[0], x[1])):
                model.add_new_mask(state, j, oid, mask)
                seed_meta.append(dict(frame=j, object_id=oid, area=int(mask.sum()), **meta))
            for r in local_extra:
                j = next(j for j, (idx, _) in enumerate(clip_samples) if idx == r['ui_index'])
                s = clip_samples[j][1]
                if r.get('empty'):
                    model.add_new_mask(state, j, r['object_id'], np.zeros((s['image_height'], s['image_width']), bool))
                elif 'mask_file' in r:
                    model.add_new_mask(state, j, r['object_id'], np.load(WORK / r['mask_file']).astype(bool))
                else:
                    scale = np.array([s['image_width'], s['image_height']])
                    points = np.asarray(r['points'], np.float32) * scale if r.get('points') else None
                    labels = np.asarray(r.get('labels', [1] * len(points)), np.int32) if points is not None else None
                    box = np.asarray(r['box'], np.float32) * np.tile(scale, 2) if 'box' in r else None
                    model.add_new_points_or_box(state, j, r['object_id'], points=points, labels=labels, box=box)
                seed_meta.append(dict(frame=j, source='visual_prompt', **r))
            earliest = min(r['frame'] for r in seed_meta)
            output = {}
            for reverse in [False, True] if earliest else [False]:
                for j, object_ids, logits in model.propagate_in_video(state, start_frame_idx=earliest, reverse=reverse):
                    masks = (logits[:, 0] > 0).cpu().numpy()
                    output[j] = dict(zip(object_ids, masks))
                    if j % 5 == 0:
                        print('FRAME', clip_no, j, 'areas', [int(a.sum()) for a in masks], 'sec', round(time.time()-started, 1), flush=True)
            ids = sorted(positive)
            packed = np.stack([np.stack([output[j][oid] for oid in ids]) for j in range(20)])
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest, masks=packed, object_ids=np.array(ids), ui_indices=np.array([idx for idx, _ in clip_samples]))
        dest.with_suffix('.json').write_text(json.dumps({
            'clip_id': clip_id, 'model': 'sam2.1_hiera_small', 'device': args.device,
            'checkpoint_sha256': hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            'seed': 20260916, 'seeds': seed_meta, 'seconds': time.time()-started,
            'areas': packed.sum(axis=(2, 3)).tolist(), 'object_ids': ids,
        }, ensure_ascii=False, indent=2))
        print('DONE', clip_no, 'sec', round(time.time()-started, 1), flush=True)
        del state, packed, output, logits, masks
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
