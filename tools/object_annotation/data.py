"""Read-only adapters. A manifest decouples annotation from storage layout."""
import json
from pathlib import Path
import random
import re

import cv2
import numpy as np
import pyarrow.parquet as pq

CAMERAS = {'head':'head', 'left_wrist':'left', 'right_wrist':'right'}


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)',str(path))]


def scan(root):
    root = Path(root).resolve()
    if root.is_file():
        data = json.loads(root.read_text())
        episodes = data['episodes'] if isinstance(data,dict) else data
        for ep in episodes:
            assert set(ep['cameras']) == set(CAMERAS), 'Manifest must name all three cameras'
        return episodes
    manifest = root/'meta/source_manifest.json'
    if manifest.exists():
        info = json.loads((root/'meta/info.json').read_text())
        result = []
        for ep in json.loads(manifest.read_text())['episodes']:
            idx = ep['episode_index']; chunk = idx//info['chunks_size']
            item = dict(episode_id=f'episode_{idx:06d}', episode_index=idx,
                        task_id=ep['source_task_number'], side=ep['source_side'],
                        source_episode_number=ep['source_episode_number'], frames=ep['frames'],
                        source_hdf5=ep['source'], fps=info['fps'],
                        parquet=str(root/info['data_path'].format(episode_index=idx,episode_chunk=chunk)),
                        source_timestamp_start=ep['source_timestamp_start'], cameras={})
            for camera, key in CAMERAS.items():
                item['cameras'][camera] = dict(video=str(root/info['video_path'].format(
                    episode_index=idx,episode_chunk=chunk,video_key=f'observation.images.{key}')),
                    timestamp_column=f'timestamp.images.{key}')
            result.append(item)
        return result
    # Convenience image-folder adapter. Arbitrary layouts use explicit manifests.
    result = []
    for head in sorted(root.rglob('head'),key=natural_key):
        epdir = head.parent
        task = next((re.fullmatch(r'task_(\d+)',p.name) for p in epdir.parents
                     if re.fullmatch(r'task_(\d+)',p.name)),None)
        if task is None: continue
        cameras = {}
        for camera in CAMERAS:
            files = sorted([p for p in (epdir/camera).glob('*')
                            if p.suffix.lower() in ('.jpg','.jpeg','.png')],key=natural_key)
            if not files: raise ValueError(f'Missing image stream: {epdir/camera}')
            cameras[camera] = dict(images=[str(p) for p in files])
        result.append(dict(episode_id=epdir.name,task_id=int(task[1]),cameras=cameras,fps=30))
    if not result: raise ValueError('No supported dataset found; provide a JSON episode manifest')
    return result


def select(episodes, tasks, count, seed):
    rng = random.Random(seed); chosen=[]
    for task in tasks:
        subset = [e for e in episodes if e['task_id']==task]
        if not subset: raise ValueError(f'No episodes for task {task}')
        if not count:
            chosen.extend(subset); continue
        if len(subset)<count: raise ValueError(f'Only {len(subset)} episodes for task {task}')
        if all(e.get('side') in ('left','right') for e in subset):
            # Deliberately include both mirrored variants, reproducibly.
            for side,n in [('left',(count+1)//2),('right',count//2)]:
                chosen.extend(rng.sample([e for e in subset if e['side']==side],n))
        else: chosen.extend(rng.sample(subset,count))
    ids=[(e['task_id'],e['episode_id']) for e in chosen]
    assert len(ids)==len(set(ids)), 'Non-unique output episode IDs'
    for e in chosen:
        assert re.fullmatch(r'[A-Za-z0-9_-]+',e['episode_id']), 'Unsafe episode ID'
        for cam in e['cameras'].values():
            paths=[cam['video']] if 'video' in cam else cam['images']
            for path in paths:
                if not Path(path).is_file(): raise FileNotFoundError(path)
    return sorted(chosen,key=lambda e:(e['task_id'],e['episode_id']))


def frames(ep,camera,stride=1,max_frames=0):
    if stride<1 or max_frames<0:raise ValueError('Invalid stride/frame limit')
    spec=ep['cameras'][camera]; times=spec.get('timestamps')
    if 'parquet' in ep:
        times = pq.read_table(ep['parquet'],columns=[spec['timestamp_column']])[spec['timestamp_column']].to_numpy()
        times = times + ep['source_timestamp_start']
        assert len(times)==ep['frames']
    cap=cv2.VideoCapture(spec['video']) if 'video' in spec else None
    if cap is not None and not cap.isOpened(): raise IOError(spec['video'])
    n=0; yielded=0
    try:
        while True:
            if cap is not None:
                ok=cap.grab()
                if not ok: break
                if n%stride:
                    n+=1; continue
                ok,image=cap.retrieve()
                if not ok: raise IOError(f'Cannot decode frame {n}')
            else:
                if n>=len(spec['images']): break
                if n%stride:
                    n+=1; continue
                image=cv2.imread(spec['images'][n])
                if image is None: raise IOError(spec['images'][n])
            ts=None if times is None else float(times[n])
            if ts is not None and not np.isfinite(ts): raise ValueError('Nonfinite timestamp')
            yield n,ts,image
            n+=1; yielded+=1
            if max_frames and yielded>=max_frames: return
        if 'frames' in ep and n!=ep['frames']:
            raise IOError(f'Truncated video: decoded {n}, expected {ep["frames"]}')
        if times is not None and n!=len(times):
            raise ValueError('Timestamp count differs from the actual stream length')
        if n==0:raise ValueError('Empty camera stream')
    finally:
        if cap is not None: cap.release()
