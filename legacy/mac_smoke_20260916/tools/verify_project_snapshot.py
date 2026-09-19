"""Read-only verification of the confirmed project; Python standard library only.

Run from any directory. If PACKAGE_MANIFEST.json exists at the project root,
verify every archived file too. No model, browser, server, or GPU is needed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'astribot_mask_audit_420_20260916'
AUDIT = BUNDLE / 'audit'
sys.dont_write_bytecode = True  # Keep first-run verification entirely read-only.
sys.path.insert(0, str(BUNDLE))
from audit_rle import decode_counts, validate_counts


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional report path; otherwise prints only')
    args = parser.parse_args()
    manifest_path = AUDIT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    bundle_info = json.loads((BUNDLE / 'bundle_info.json').read_text())
    assert sha(manifest_path) == bundle_info['audit_manifest_sha256']
    samples = {s['sample_id']: s for s in manifest['samples']}
    assert len(samples) == len(manifest['samples']) == 420
    for sample in samples.values():
        path = AUDIT / sample['image']
        assert sha(path) == sample['image_sha256'], path
        with path.open('rb') as f:
            header = f.read(24)
        assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
        assert struct.unpack('>II', header[16:24]) == (sample['image_width'], sample['image_height'])
    humans, proposals, visibility, areas = {}, {}, Counter(), Counter()
    expected = {(s['sample_id'], o['object_id']) for s in samples.values() for o in s['objects']}
    for folder, rows in [('human_labels', humans), ('proposed_labels', proposals)]:
        for path in (AUDIT / folder).glob('*.json'):
            row = json.loads(path.read_text())
            key = row['sample_id'], row['object_id']
            assert key in expected and key not in rows, path
            sample = samples[key[0]]
            assert path.name == f'{key[0]}__{key[1]}.json'
            assert row['source_image_sha256'] == sample['image_sha256']
            assert row['mask']['size'] == [sample['image_height'], sample['image_width']]
            assert row['instance_id'].strip()
            assert set(row['conditions']) <= set(manifest['required_conditions'])
            assert row['visibility'] in ['visible', 'partial_occlusion', 'fully_occluded', 'out_of_view', 'unknown']
            area = validate_counts(decode_counts(row['mask']), sample['image_height'], sample['image_width'])
            assert bool(area) == (row['visibility'] in ['visible', 'partial_occlusion']), path
            if folder == 'human_labels':
                assert row['human_confirmed'] is True and row['annotator'].strip()
                assert row['annotation_method'] == 'manual_pixel_editor'
                if row.get('prediction_prefill'):
                    assert row.get('proposal_provenance', {}).get('proposal_id')
                visibility[row['visibility']] += 1
                areas['nonempty' if area else 'empty'] += 1
            else:
                assert row['human_confirmed'] is False and row['prediction_prefill'] is True
            rows[key] = row
    assert set(humans) == expected and len(humans) == 1860
    assert set(proposals) <= set(humans)  # Historical proposals remain after approval.
    revisions = (AUDIT / 'human_label_revisions.jsonl').read_bytes()
    latest, previous_revision = {}, {}
    for line in revisions.splitlines():
        row = json.loads(line)
        key = row['sample_id'], row['object_id']
        assert key in expected and row['human_confirmed'] is True
        assert row['revision'] > previous_revision.get(key, 0)
        previous_revision[key] = row['revision']; latest[key] = row
    assert latest == humans, 'Latest revision differs from current human label'
    with zipfile.ZipFile(ROOT / 'handoffs/human_labels_before_assistance_20260916_204158.zip') as z:
        for name in z.namelist():
            if name == 'backup_info.json':
                continue
            if name.endswith('human_label_revisions.jsonl'):
                assert revisions.startswith(z.read(name))
            else:
                assert (BUNDLE / name).read_bytes() == z.read(name)
    checkpoint = ROOT / 'models/sam2.1_hiera_small.hf.pt'
    assert sha(checkpoint) == '6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
    package_manifest = ROOT / 'PACKAGE_MANIFEST.json'
    package_count = None
    if package_manifest.exists():
        package = json.loads(package_manifest.read_text())
        for entry in package['files']:
            path = ROOT / entry['path']
            assert path.resolve().is_relative_to(ROOT.resolve()), entry['path']
            assert path.stat().st_size == entry['size'], entry['path']
            assert sha(path) == entry['sha256'], entry['path']
        package_count = len(package['files'])
    report = dict(result='PASS', frames=len(samples), confirmed=len(humans), pending=0,
                  complete_frames=420, archived_proposals=len(proposals),
                  visibility=dict(visibility), masks=dict(areas), revisions=len(revisions.splitlines()),
                  images='all 420 hashes and PNG dimensions match',
                  original_78_human_labels='unchanged', original_revision_history='preserved as prefix',
                  latest_revisions='match all current human labels', checkpoint_sha256=sha(checkpoint),
                  package_files_verified=package_count)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
