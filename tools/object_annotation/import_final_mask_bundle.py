"""Verify a final reviewed image/mask ZIP and create a versioned accepted view."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image

from masks import bbox, decode, encode


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', required=True, type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    assert not args.output.exists(), 'Use a new destination; never overwrite accepted labels'
    assert sha(args.zip.read_bytes()) == args.sha256
    reference = {s['case_id']: s for s in json.loads((args.reference/'manifest.json').read_text())['samples']}
    rows, visibility, stale, overlaps = [], Counter(), Counter(), []
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in name
            assert not archive.getinfo(name).is_dir()
            assert archive.getinfo(name).external_attr >> 16 & 0o170000 != 0o120000
        assert archive.testzip() is None
        hashes = {}
        for line in archive.read('SHA256SUMS').decode().splitlines():
            digest, name = line.split(None, 1)
            name = name.lstrip('*')
            assert name not in hashes
            assert sha(archive.read(name)) == digest, name
            hashes[name] = digest
        assert set(hashes) == set(names)-{'SHA256SUMS'}
        manifest = json.loads(archive.read('manifest.json'))
        samples = manifest['samples']
        assert len(samples) == len(reference) == 87
        assert {s['sample_id'] for s in samples} == set(reference)
        mapping = json.loads(archive.read('object_id_mapping.json'))
        mapped = {(m['case_id'], m['final_object_id']): m for m in mapping}
        assert len(mapped) == len(mapping)
        expected_files = set()
        for sample in samples:
            key = sample['sample_id']
            ref = reference[key]
            assert all(sample[k] == ref[k] for k in ('episode_id', 'task_id', 'camera', 'frame_idx'))
            image = archive.read(sample['image'])
            assert sha(image) == sample['image_sha256'] == ref['image_sha256']
            with Image.open(io.BytesIO(image)) as im:
                im.load()
                assert im.size == (sample['image_width'], sample['image_height'])
            objects = {o['object_id']: o for o in sample['objects']}
            assert len(objects) == len(sample['objects']) and 1100 not in objects
            assert {1000, 1101, 1102, 1103, 1104} <= set(objects)
            assert set(sample['final_mask_files']) == {f'masks/{key}__{i}.json' for i in objects}
            binary = {}
            raw_rows = {}
            for oid, obj in objects.items():
                file = f'masks/{key}__{oid}.json'
                expected_files.add(file)
                row = json.loads(archive.read(file))
                assert row['sample_id'] == key and row['object_id'] == oid
                assert all(row[k] == sample[k] for k in ('episode_id', 'camera', 'frame_idx'))
                assert row['human_confirmed'] is True
                assert row['source_image_sha256'] == sample['image_sha256']
                assert (key, oid) in mapped
                assert mapped[key, oid].get('merged_object_ids') == obj.get('merged_object_ids')
                assert mapped[key, oid].get('derived_from') == obj.get('derived_from')
                mask = decode(row['mask'])
                assert mask.shape == (sample['image_height'], sample['image_width'])
                assert row['visibility'] in {'visible', 'partial_occlusion', 'out_of_view', 'not_visible', 'ignored_blur'}
                assert mask.any() == (row['visibility'] in {'visible', 'partial_occlusion'})
                binary[oid], raw_rows[oid] = mask, row
                visibility[row['visibility']] += 1
                stale[str(row.get('review_status'))] += 1
            union = np.logical_or.reduce([binary[i] for i in (1101, 1102, 1103, 1104)])
            assert np.array_equal(binary[1000], union), f'Robot union mismatch: {key}'
            overlap = int((np.stack([binary[i] for i in (1101, 1102, 1103, 1104)]).sum(0)>1).sum())
            if overlap:
                overlaps.append(dict(sample_id=key, overlapping_part_pixels=overlap))
            original_case = json.loads((args.reference/key/'case.json').read_text())
            timestamp = original_case['predictions']['objects'][0].get('timestamp')
            for oid, obj in objects.items():
                source = raw_rows[oid]
                mask = union if oid == 1000 else binary[oid]
                ignored = source['visibility'] == 'ignored_blur'
                rows.append(dict(sample_id=key, episode_id=sample['episode_id'], task_id=sample['task_id'],
                    camera=sample['camera'], frame_idx=sample['frame_idx'], timestamp=timestamp,
                    object_id=oid, class_name=obj['class_name'], group=obj['group'],
                    merged_object_ids=obj.get('merged_object_ids'),
                    semantic_class='meat' if obj.get('merged_object_ids') else obj['class_name'],
                    bbox_xyxy=bbox(mask), mask=encode(mask), mask_area=int(mask.sum()),
                    visible=None if ignored else bool(mask.any()), visibility=source['visibility'],
                    exclude_from_training=ignored, exclude_from_metrics=ignored,
                    confidence=None, human_confirmed=True, review_status='human_confirmed_final',
                    image_width=sample['image_width'], image_height=sample['image_height'],
                    provenance=dict(archive_sha256=args.sha256, source_label=f'package/masks/{key}__{oid}.json',
                        source_label_sha256=hashes[f'masks/{key}__{oid}.json'],
                        source_image_sha256=sample['image_sha256'],
                        original_review_status=source.get('review_status'),
                        annotation_method=source.get('annotation_method'),
                        acceptance='User final confirmation 2026-09-18 and DATASET_INFO.json',
                        derived_robot_union=oid==1000)))
        assert expected_files == {n for n in names if n.startswith('masks/')}
        assert len(rows) == len(mapped) == 654
        args.output.mkdir(parents=True)
        archive.extractall(args.output/'package')
    report = dict(result='PASS', archive=str(args.zip.resolve()), archive_sha256=args.sha256,
        frames=87, editable_labels=567, derived_robot_labels=87, mask_records=654,
        verified_internal_hashes=len(hashes), visibility=dict(visibility),
        source_review_status_histogram=dict(stale), part_overlap_frames=overlaps,
        warnings=['Historical pending review_status fields retained in immutable package; final user confirmation takes precedence.',
                  'High-risk selected set is not an unbiased corpus accuracy sample.'],
        checks=['ZIP CRC and SHA256', 'All manifest-listed file hashes', '87 exact source image/frame matches',
                '654 mask RLE decodes/shapes/visibility', 'No 1100', '87 exact four-part robot unions',
                'Merged IDs follow manifest', 'ignored_blur excluded from training/metrics'],
        production_masks_changed=False, original_package_preserved=True)
    (args.output/'accepted_masks.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (args.output/'import_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
