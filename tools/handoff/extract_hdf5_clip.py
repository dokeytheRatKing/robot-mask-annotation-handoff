"""Extract an explicit source-frame window to a versioned RGB-only clip."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import save, sha


def decode_rgb(value, channel_order):
    if isinstance(value, np.ndarray) and value.ndim == 3:
        if value.dtype != np.uint8 or value.shape[-1] != 3:
            raise ValueError('Raw RGB must be uint8 H,W,3; do not infer scaling or alpha semantics')
        return value if channel_order == 'RGB' else value[..., ::-1]
    if isinstance(value, (bytes, np.bytes_)):
        raw = bytes(value)
    elif isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.uint8:
        raw = value.tobytes()
    else:
        raise ValueError('Expected uint8 image or encoded JPEG/PNG bytes')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not decode selected frame')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    import h5py
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--rgb-key', required=True)
    p.add_argument('--timestamp-key')
    p.add_argument('--channel-order', choices=['RGB', 'BGR'], required=True)
    p.add_argument('--task-id', required=True)
    p.add_argument('--episode-id', required=True)
    p.add_argument('--camera', required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert a.start >= 0 and 0 < a.count <= 128 and a.stride > 0
    source, dest = a.input.resolve(), a.output.resolve()
    if dest == source or dest.is_relative_to(source.parent):
        raise ValueError('Output must be separate from source dataset directory')
    dest.mkdir(parents=True, exist_ok=False)
    (dest/'rgb').mkdir()
    cv2.setNumThreads(1)
    before = source.stat()
    source_hash = sha(source)
    rows = []
    shape = None
    with h5py.File(source, 'r') as f:
        data = f[a.rgb_key]
        source_frames = len(data)
        ids = list(range(a.start, a.start+a.count*a.stride, a.stride))
        assert ids[-1] < len(data), 'Requested source range exceeds actual camera stream'
        times = f[a.timestamp_key] if a.timestamp_key else None
        if times is not None:
            assert times.ndim == 1 and len(times) == len(data)
        for i, idx in enumerate(ids):
            rgb = decode_rgb(data[idx], a.channel_order)
            if shape is None:
                shape = list(rgb.shape[:2])
            assert list(rgb.shape[:2]) == shape
            image = dest/'rgb'/f'{i:05d}.png'
            Image.fromarray(rgb).save(image)
            timestamp = float(times[idx]) if times is not None else None
            assert timestamp is None or np.isfinite(timestamp)
            rows.append(dict(local_frame_idx=i, source_frame_idx=idx, timestamp=timestamp,
                image='rgb/'+image.name, image_sha256=sha(image)))
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source changed during extraction'
    result = dict(schema='mask_handoff.clip.v1', clip_id=dest.name, task_id=a.task_id,
        episode_id=a.episode_id, camera=a.camera, image_shape=shape, frames=rows, object_ids=[],
        provenance=dict(source_filename=source.name, source_sha256=source_hash, rgb_key=a.rgb_key,
            timestamp_key=a.timestamp_key, channel_order=a.channel_order, source_frames=source_frames,
            action_state_untouched=True, task_registry_status='must_be_bound_before_annotation'))
    save(dest/'clip.json', result)
    print(json.dumps(dict(output=str(dest), frames=len(rows), source_sha256=source_hash), indent=2))


if __name__ == '__main__':
    main()
