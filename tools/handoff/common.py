"""Portable read-only clip validation; no Torch imports."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def resolve_under(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    dest = (root / rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or not dest.is_relative_to(root):
        raise ValueError('Expected a relative path inside clip root')
    return dest


def load_clip(path, max_frames=128):
    from PIL import Image
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema') != 'mask_handoff.clip.v1':
        raise ValueError('Unsupported clip schema')
    frames = data['frames']
    if not 0 < len(frames) <= max_frames:
        raise ValueError('Clip exceeds explicit bounded frame limit')
    source_ids = []
    for i, row in enumerate(frames):
        assert row['local_frame_idx'] == i
        assert type(row['source_frame_idx']) is int
        source_ids.append(row['source_frame_idx'])
        rgb = resolve_under(path.parent, row['image'])
        assert sha(rgb) == row['image_sha256'], rgb
        with Image.open(rgb) as image:
            assert list(image.size[::-1]) == data['image_shape']
            assert image.mode == 'RGB'
    assert source_ids == sorted(set(source_ids))
    return path, data


def load_mask_seeds(path, clip):
    import numpy as np
    from pycocotools import mask as coco
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    assert data['schema'] == 'mask_handoff.seeds.v1' and data['clip_id'] == clip['clip_id']
    if not (data.get('reviewed_example') or data.get('visual_review')):
        raise ValueError('Record actual seed overlay inspection before propagation')
    seen = set()
    for seed in data['seeds']:
        i, oid = seed['local_frame_idx'], seed['object_id']
        assert type(i) is int and 0 <= i < len(clip['frames']) and type(oid) is int
        assert (i, oid) not in seen
        seen.add((i, oid))
        row = clip['frames'][i]
        assert seed['source_frame_idx'] == row['source_frame_idx']
        assert seed['source_image_sha256'] == row['image_sha256']
        if not seed.get('mask'):
            raise ValueError('This runner accepts reviewed positive RLE seeds. Generate and inspect point/box proposals separately.')
        rle = seed['mask']
        assert rle['format'] == 'coco_rle' and rle['size'] == clip['image_shape']
        binary = coco.decode(dict(size=rle['size'], counts=rle['counts'].encode('ascii'))).astype(bool)
        assert binary.any(), 'Empty/unknown seeds need explicit interval handling, not global absence propagation'
        assert binary.shape == tuple(clip['image_shape']) and np.isfinite(binary).all()
    assert seen
    return path, data
