"""Bounded portable SAM2 runner from reviewed positive RLE seeds; drafts only."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

from common import load_clip, load_mask_seeds, save, sha

ROOT = Path(__file__).resolve().parents[2]


def intervals(seeds, nframes):
    by_object = defaultdict(list)
    for seed in seeds:
        by_object[seed['object_id']].append(seed)
    for oid, entries in sorted(by_object.items()):
        entries.sort(key=lambda r: r['local_frame_idx'])
        for i, seed in enumerate(entries):
            t = seed['local_frame_idx']
            left = 0 if i == 0 else (entries[i-1]['local_frame_idx']+t)//2+1
            right = nframes-1 if i == len(entries)-1 else (t+entries[i+1]['local_frame_idx'])//2
            yield oid, seed, left, right


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', type=Path, required=True)
    p.add_argument('--seeds', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'models/sam2/sam2.1_hiera_base_plus.pt')
    p.add_argument('--model-config', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=128)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    path, clip = load_clip(a.clip, a.max_frames)
    seed_path, seed_spec = load_mask_seeds(a.seeds, clip)
    dest = a.output.resolve()
    if dest == path.parent or dest.is_relative_to(path.parent) or dest.is_relative_to(ROOT/'examples'):
        raise ValueError('Output must be outside input clip/source examples')
    if dest.exists():
        raise FileExistsError(dest)
    spec = dict(status='dry_run' if a.dry_run else 'started', clip_id=clip['clip_id'],
        frames=len(clip['frames']), seed_count=len(seed_spec['seeds']),
        planned_intervals=[dict(object_id=o, seed_local=s['local_frame_idx'], left=l, right=r)
            for o, s, l, r in intervals(seed_spec['seeds'], len(clip['frames']))],
        clip_sha256=sha(path), seeds_sha256=sha(seed_path), script_sha256=sha(__file__),
        schema='mask_handoff.propagation.v1', output_kind='unconfirmed_draft',
        offline_future_seeds_allowed=True, model_training=False)
    if a.dry_run:
        print(json.dumps(spec, indent=2))
        return
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or visible == '-1':
        raise ValueError('Explicitly select one verified available GPU with CUDA_VISIBLE_DEVICES')
    if not a.checkpoint.is_file():
        raise FileNotFoundError(a.checkpoint)
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    sys.path.insert(0, str(ROOT/'tools/object_annotation'))
    from masks import decode, record
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    dest.mkdir(parents=True)
    cache = dest/'sam_frames'
    cache.mkdir()
    for frame in clip['frames']:
        with Image.open(path.parent/frame['image']) as im:
            im.save(cache/f'{frame["local_frame_idx"]:05d}.jpg', quality=95)
    spec.update(checkpoint_sha256=sha(a.checkpoint), model_config=a.model_config,
        torch_version=torch.__version__, visible_gpu=visible, working_rgb='derived JPEG95; original PNG immutable')
    save(dest/'run.json', spec)
    start = time.monotonic()
    predictor = build_sam2_video_predictor(a.model_config, str(a.checkpoint), device='cuda',
        apply_postprocessing=False,
        hydra_overrides_extra=['++model.fill_hole_area=0', '++model.non_overlap_masks=false'])
    predictor.eval().requires_grad_(False)
    output = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for oid, seed, left, right in intervals(seed_spec['seeds'], len(clip['frames'])):
            index = seed['local_frame_idx']
            for reverse, end in [(True, left), (False, right)]:
                state = predictor.init_state(str(cache), offload_video_to_cpu=True, offload_state_to_cpu=True)
                predictor.add_new_mask(state, index, oid, decode(seed['mask']))
                for local, ids, logits in predictor.propagate_in_video(state, start_frame_idx=index,
                        max_frame_num_to_track=abs(end-index), reverse=reverse):
                    if not left <= local <= right:
                        raise ValueError('SAM output escaped assigned interval')
                    assert list(ids) == [oid]
                    output[(local, oid)] = (logits[0, 0] > 0).cpu().numpy(), seed
                del state
            # Exact seed pixels are immutable regardless of predictor output.
            output[(index, oid)] = decode(seed['mask']), seed
    expected = {(i, s['object_id']) for i in range(len(clip['frames'])) for s in seed_spec['seeds']}
    assert set(output) == expected
    rows = []
    for (i, oid), (binary, seed) in sorted(output.items()):
        frame = clip['frames'][i]
        exact = i == seed['local_frame_idx']
        row = dict(clip_id=clip['clip_id'], task_id=clip['task_id'], episode_id=clip['episode_id'],
            camera=clip['camera'], local_frame_idx=i, frame_idx=frame['source_frame_idx'],
            timestamp=frame['timestamp'], object_id=oid, source_image_sha256=frame['image_sha256'],
            human_confirmed=exact and bool(seed.get('human_confirmed', False)),
            training_eligible=False, confidence=None, review_status='pending_nonseed_visual_review',
            provenance=dict(kind='exact_seed' if exact else 'offline_sam2_propagation',
                seed_local_frame_idx=seed['local_frame_idx'], seed_source_frame_idx=seed['source_frame_idx'],
                seed_spec_sha256=sha(seed_path), future_seed=seed['local_frame_idx'] > i))
        if binary.any():
            row.update(record(binary), visibility='visible_seed' if exact else 'predicted_visible')
        else:
            row.update(mask=None, mask_area=None, bbox_xyxy=None, visible=None, visibility='unknown', mask_status='unknown')
        rows.append(row)
    (dest/'draft.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))
    save(dest/'complete.json', dict(status='engineering_complete_semantic_review_pending', rows=len(rows),
        seconds=time.monotonic()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        draft_sha256=sha(dest/'draft.jsonl'), semantic_acceptance=False, human_confirmation_propagated=False))
    print(json.dumps(dict(output=str(dest), rows=len(rows), semantic_acceptance=False), indent=2))


if __name__ == '__main__':
    main()
