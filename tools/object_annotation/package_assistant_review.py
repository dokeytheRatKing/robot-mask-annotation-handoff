"""Package manually authored visual decisions without altering production masks."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--zip', required=True, type=Path)
    args = parser.parse_args()
    root = args.review.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    samples = {r['case_id']: r for r in manifest['samples']}
    with (root/'visual_decisions.tsv').open() as handle:
        decisions = list(csv.DictReader(handle, delimiter='\t'))
    assert len(decisions) == len(samples) == len({d['case_id'] for d in decisions})
    assert {d['case_id'] for d in decisions} == set(samples)
    cv2.setNumThreads(1)
    counts = Counter()
    for decision in decisions:
        key = decision['case_id']
        sample = samples[key]
        folder = root/key
        assert hashlib.sha256((folder/'rgb.png').read_bytes()).hexdigest() == sample['image_sha256']
        decision.update({k: sample[k] for k in ('episode_id', 'camera', 'frame_idx', 'task_id')})
        decision.update(reviewer='assistant_multimodal_visual_review', human_confirmed=False,
                        human_verdict='', human_notes='', annotation_modified=False)
        counts[decision['status']] += 1
        if decision['status'] == 'needs_human_review':
            cap = cv2.VideoCapture(sample['source_video'])
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                panels = []
                for delta in (-15, 0, 15):
                    index = min(total-1, max(0, sample['frame_idx']+delta))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    assert ok, (key, index)
                    panel = cv2.resize(frame, (640, 360))
                    panel = np.pad(panel, ((32, 0), (0, 0), (0, 0)))
                    cv2.putText(panel, f'{key} frame {index}', (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                    panels.append(panel)
                assert cv2.imwrite(str(folder/'context.jpg'), np.concatenate(panels, axis=1))
            finally:
                cap.release()
        sample['assistant_review_status'] = decision['status']
    report = dict(scope=manifest['scope'], counts=dict(counts), cases=decisions,
                  limitations=['High-risk selected frames are not an accuracy sample.',
                               'No human GT or pixel IoU is established by this review.',
                               'Single-frame review does not establish temporal recovery.',
                               'No production predictions were changed.'])
    (root/'review_results.json').write_text(json.dumps(report, indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with (root/'review.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    with zipfile.ZipFile(args.zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                archive.write(path, arcname=Path(root.name)/path.relative_to(root))
    with zipfile.ZipFile(args.zip) as archive:
        assert archive.testzip() is None
    digest = hashlib.sha256(args.zip.read_bytes()).hexdigest()
    args.zip.with_suffix('.zip.sha256').write_text(f'{digest}  {args.zip.name}\n')
    print(json.dumps(dict(counts=counts, zip=str(args.zip), bytes=args.zip.stat().st_size,
                          sha256=digest), indent=2))


if __name__ == '__main__':
    main()
