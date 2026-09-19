"""Export reproducible, read-only task/camera strata for actual visual review."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from annotate import write_json
from full_segmentation import render


def read_rows(path):
    with gzip.open(path, 'rt') as handle:
        yield from map(json.loads, handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    episodes = {e['episode_id']: e for e in json.loads((root/'episodes.json').read_text())['episodes']}
    with sqlite3.connect(f'file:{root}/queue.sqlite?mode=ro', uri=True) as db:
        jobs = [json.loads(r[0]) for r in db.execute("SELECT result FROM jobs WHERE status='done'")]
    manifest = []
    sheets = {}
    for task in sorted({j['task_id'] for j in jobs}):
        for camera in ('head', 'left_wrist', 'right_wrist'):
            candidates = [j for j in jobs if j['task_id'] == task and j['camera'] == camera]
            if not candidates:
                continue
            # High-flag strata are deliberately biased; not an accuracy estimate.
            job = max(candidates, key=lambda j: (j['review_events']/j['frames'], j['episode_id']))
            events = list(read_rows(root/job['files']['review']))
            def score(event):
                reasons = [reason for issue in event['issues'] for reason in issue['reasons']]
                return (sum(3 if 'overlap' in s or 'jump' in s else 1 for s in reasons),
                        -abs(event['frame_idx']-job['frames']/2))
            event = max(events, key=score) if events else {'frame_idx': job['frames']//2, 'issues': []}
            idx = event['frame_idx']
            selected = {}
            for layer in ('objects', 'robot_parts', 'robot'):
                selected[layer] = []
                for row in read_rows(root/job['files'][layer]):
                    if row['frame_idx'] == idx:
                        selected[layer].append(row)
                    if row['frame_idx'] > idx:
                        break
            episode = episodes[job['episode_id']]
            video = episode['cameras'][camera]['video']
            cap = cv2.VideoCapture(video)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, image = cap.read()
                assert ok, (video, idx)
            finally:
                cap.release()
            case = f'T{task:02d}_{camera}'
            folder = args.output/case
            folder.mkdir()
            assert cv2.imwrite(str(folder/'rgb.png'), image)
            objects = render(image, selected['objects'], None, '')
            robot = render(image, [], selected['robot'][0], '', selected['robot_parts'])
            preview = np.concatenate([objects[30:, :640], objects[30:, 640:], robot[30:, 640:]], axis=1)
            preview = np.pad(preview, ((40, 0), (0, 0), (0, 0)))
            title = f'{case} {job["episode_id"]} frame {idx} | RGB / OBJECTS / ROBOT (unconfirmed)'
            cv2.putText(preview, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
            assert cv2.imwrite(str(folder/'comparison.jpg'), preview)
            sheets.setdefault(task, []).append(preview)
            record = dict(case_id=case, task_id=task, episode_id=job['episode_id'], camera=camera,
                frame_idx=idx, source_video=video, source_attempt=job['attempt'],
                source_files=job['files'], source_hashes=job['file_hashes'],
                automatic_issues=event['issues'], predictions=selected,
                image_sha256=hashlib.sha256((folder/'rgb.png').read_bytes()).hexdigest(),
                assistant_review_status='pending', human_confirmed=False)
            write_json(folder/'case.json', record)
            manifest.append({k:v for k,v in record.items() if k != 'predictions'})
            print(case, job['episode_id'], idx, flush=True)
        assert cv2.imwrite(str(args.output/f'task_{task:02d}.jpg'), np.concatenate(sheets[task], axis=0))
    write_json(args.output/'manifest.json', dict(source_root=str(root), samples=manifest,
        scope='One high-risk frame per completed task/camera stratum; not exhaustive or random.',
        policy='Export is not visual review. Decisions require actual RGB inspection; never auto-confirm.'))


if __name__ == '__main__':
    main()
