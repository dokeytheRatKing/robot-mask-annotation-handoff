"""Check full label coverage, original inputs, RLE and human-label preservation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image
from assist_masks import ROOT, AUDIT, WORK, encode, decode
sys.path.insert(0, str(AUDIT.parent))
from audit_rle import decode_counts, validate_counts


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest_raw = (AUDIT / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    assert sha(manifest_raw) == json.loads((AUDIT.parent / 'bundle_info.json').read_text())['audit_manifest_sha256']
    backup = ROOT / 'handoffs/human_labels_before_assistance_20260916_204158.zip'
    with zipfile.ZipFile(backup) as z:
        for name in z.namelist():
            if name == 'backup_info.json':
                continue
            path = AUDIT.parent / name
            assert path.read_bytes() == z.read(name), f'Changed human input: {path}'
    humans = {p.name:json.loads(p.read_text()) for p in (AUDIT / 'human_labels').glob('*.json')}
    proposals = {p.name:json.loads(p.read_text()) for p in (AUDIT / 'proposed_labels').glob('*.json')}
    expected, visibility, sources, areas = set(), Counter(), Counter(), Counter()
    proposal_overlaps = []
    for i, s in enumerate(manifest['samples'], 1):
        path = AUDIT / s['image']
        assert sha(path.read_bytes()) == s['image_sha256']
        with Image.open(path) as im:
            assert im.size == (s['image_width'], s['image_height'])
        masks, is_human = [], []
        for obj in s['objects']:
            name = f'{s["sample_id"]}__{obj["object_id"]}.json'
            expected.add(name)
            human = name in humans
            row = humans[name] if human else proposals[name]
            assert row['sample_id'] == s['sample_id'] and row['object_id'] == obj['object_id']
            assert row['source_image_sha256'] == s['image_sha256']
            assert row['mask']['size'] == [s['image_height'], s['image_width']]
            assert row['human_confirmed'] is human
            assert set(row['conditions']) <= set(manifest['required_conditions'])
            counts = decode_counts(row['mask'])
            area = validate_counts(counts, s['image_height'], s['image_width'])
            assert bool(area) == (row['visibility'] in ['visible','partial_occlusion'])
            mask = decode(row['mask'])
            assert int(mask.sum()) == area
            assert decode_counts(encode(mask)) == counts
            if not human:
                assert row['review_status'] == 'pending_human_review' and row['prediction_prefill'] is True
                assert row['annotator'] == '' and row['mask_area'] == area
                assert not ('normal_unoccluded' in row['conditions'] and 'partial_occlusion' in row['conditions'])
                assert ('out_of_view' in row['conditions']) == (row['visibility']=='out_of_view')
            masks.append(mask);is_human.append(human)
            visibility[row['visibility']] += 1
            sources['human' if human else 'proposal'] += 1
            areas['nonempty' if area else 'empty'] += 1
        for j in range(len(masks)):
            for k in range(j):
                if not (is_human[j] and is_human[k]):
                    assert not (masks[j] & masks[k]).any(), f'Overlapping proposal: frame {i}'
    assert set(humans).isdisjoint(proposals)
    assert set(humans) | set(proposals) == expected
    report = dict(result='PASS',frames=len(manifest['samples']), labels=len(expected),
        sources=dict(sources), visibility=dict(visibility), masks=dict(areas),
        original_images_sha256='420/420 unchanged', manifest_sha256=sha(manifest_raw),
        original_human_labels_and_revisions='byte-for-byte identical to backup',
        rle='all masks decoded, area-checked and round-tripped', proposal_overlap_pixels=0)
    (WORK / 'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
