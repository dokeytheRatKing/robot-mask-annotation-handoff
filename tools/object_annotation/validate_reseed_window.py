"""Check a re-seeded revision against its immutable parent and reviewed seeds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from annotate import sha, write_json
from masks import bbox, decode


def read_rows(path):
    result = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert row['object_id'] not in result[row['frame_idx']]
        result[row['frame_idx']][row['object_id']] = row
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads((a.root/'config.json').read_text()); sid = cfg['config']['case_id']
    complete = json.loads((a.root/'run_complete.json').read_text())
    assert complete['output_sha256'] == sha(a.root/'predictions.jsonl')
    assert cfg['parent_sha256'] == sha(a.parent/sid/'predictions.jsonl')
    assert complete['seed_manifest_sha256'] == sha(a.root/'seeds.json')
    seeds = {(r['frame_idx'], r['object_id']): r for r in json.loads((a.root/'seeds.json').read_text())}
    affected = {key[1] for key in seeds}
    old, new = read_rows(a.parent/sid/'predictions.jsonl'), read_rows(a.root/'predictions.jsonl')
    assert set(old) == set(new)
    decoded = unknown = kept = seed_matches = 0
    for idx, rows in new.items():
        assert set(rows) == set(old[idx])
        for oid, row in rows.items():
            parent = old[idx][oid]
            assert row['timestamp'] == parent['timestamp']
            assert row['human_confirmed'] == parent['human_confirmed']
            if parent['human_confirmed'] or oid not in affected:
                assert row == parent; kept += 1
            if row['mask'] is None:
                assert row['visible'] is None and row['visibility'] == 'unknown'; unknown += 1
            else:
                binary = decode(row['mask']); decoded += 1
                assert bbox(binary) == row['bbox_xyxy'] and int(binary.sum()) == row['mask_area']
                assert bool(binary.any()) == row['visible']
                if (idx, oid) in seeds:
                    assert np.array_equal(binary, decode(seeds[idx, oid]['mask'])); seed_matches += 1
            assert row['confidence'] is None or 0 <= row['confidence'] <= 1
    assert seed_matches == len(seeds)
    video = a.root/'objects.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
    cap = cv2.VideoCapture(str(video)); assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(new); cap.release()
    result = dict(status='PASS', frames=len(new), decoded_masks=decoded, unknown_labels=unknown,
                  preserved_parent_rows=kept, exact_additional_seeds=seed_matches, video_sha256=sha(video),
                  quality='Contract checks only; not pixel accuracy or re-entry success measurement.')
    write_json(a.root/'validation.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
