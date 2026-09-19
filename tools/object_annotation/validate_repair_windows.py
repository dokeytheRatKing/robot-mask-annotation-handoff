"""Exhaustive bounded repair contract checks, plus QA video decode."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from annotate import sha, write_json
from evaluate_confirmed_batch import PART_IDS
from masks import bbox, decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--accepted', type=Path, required=True)
    a = p.parse_args()
    complete = json.loads((a.root/'run_complete.json').read_text())
    config = json.loads((a.root/'run_config.json').read_text())
    accepted_path = a.accepted/'accepted_masks.jsonl'
    assert sha(accepted_path) == config['accepted_sha256']
    labels = defaultdict(dict)
    for line in accepted_path.read_text().splitlines():
        row = json.loads(line); labels[row['sample_id']][row['object_id']] = row
    decoded, unknown, seed_labels, videos = 0, 0, 0, []
    for case in complete['cases']:
        sid = case['case_id']; path = a.root/sid/'predictions.jsonl'
        assert sha(path) == case['output_sha256']
        frames = defaultdict(dict)
        for line in path.read_text().splitlines():
            row = json.loads(line); idx, oid = row['frame_idx'], row['object_id']
            assert oid not in frames[idx]
            assert oid != 1100 and oid in labels[sid]
            assert row['timestamp'] is not None and np.isfinite(row['timestamp'])
            assert row['confidence'] is None or 0 <= row['confidence'] <= 1
            assert row['human_confirmed'] == (idx == case['seed'])
            if row['mask'] is None:
                assert row['visible'] is None and row['visibility'] == 'unknown'
                assert row['bbox_xyxy'] is None and row['mask_area'] is None
                assert oid in case['unseeded_ids'] and idx != case['seed']
                unknown += 1
            else:
                binary = decode(row['mask']); decoded += 1
                assert binary.shape == (labels[sid][oid]['image_height'], labels[sid][oid]['image_width'])
                assert bbox(binary) == row['bbox_xyxy']
                assert int(binary.sum()) == row['mask_area'] and bool(binary.any()) == row['visible']
                if idx == case['seed']:
                    assert np.array_equal(binary, decode(labels[sid][oid]['mask']))
                    seed_labels += 1
            frames[idx][oid] = row
        assert set(frames) == set(range(case['start'], case['end']+1))
        for idx, rows in sorted(frames.items()):
            assert set(rows) == set(labels[sid])
            assert len({r['timestamp'] for r in rows.values()}) == 1
            union = np.zeros_like(decode(rows[1000]['mask']))
            for oid in PART_IDS:
                if rows[oid]['mask'] is not None:
                    union |= decode(rows[oid]['mask'])
            assert np.array_equal(union, decode(rows[1000]['mask']))
        timestamps = [rows[1000]['timestamp'] for _, rows in sorted(frames.items())]
        assert all(b >= a_ for a_, b in zip(timestamps, timestamps[1:]))
        for group in ('objects', 'robot_parts'):
            path = a.root/'qa'/f'{sid}_{group}.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True)
            cap = cv2.VideoCapture(str(path))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == case['frames']; cap.release()
            videos.append(dict(file=str(path.relative_to(a.root)), sha256=sha(path)))
    result = dict(status='PASS', frames=complete['frames'], decoded_masks=decoded,
                  unknown_unseeded_labels=unknown, exact_seed_labels=seed_labels, videos=videos,
                  quality='Structural contract only; no nonseed GT accuracy claim.')
    write_json(a.root/'validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
