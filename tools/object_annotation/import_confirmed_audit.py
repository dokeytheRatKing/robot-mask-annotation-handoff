"""Import a verified return without changing its snapshot or earlier experiments."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

from annotate import sha, write_json
from masks import bbox, decode


def import_audit(source, reference, output):
    source, reference, output = map(Path, (source, reference, output))
    assert not output.exists(), 'Use a new versioned destination'
    assert sha(source/'manifest.json') == sha(reference/'manifest.json'), 'Different audit manifest'
    manifest = json.loads((source/'manifest.json').read_text())
    samples = {s['sample_id']: s for s in manifest['samples']}
    expected = {(s['sample_id'], o['object_id']) for s in samples.values() for o in s['objects']}
    rows, hashes, geometry = {}, {}, {}
    for s in samples.values():
        image = Path(s['image'])
        assert not image.is_absolute() and '..' not in image.parts
        assert sha(source/image) == sha(reference/image) == s['image_sha256']
    for path in sorted((source/'human_labels').glob('*.json')):
        row = json.loads(path.read_text())
        key = row['sample_id'], row['object_id']
        assert key in expected and key not in rows
        s = samples[key[0]]
        assert path.name == f'{key[0]}__{key[1]}.json'
        assert row['human_confirmed'] is True and row['annotator'].strip()
        assert row['source_image_sha256'] == s['image_sha256']
        assert (row['episode_id'], row['camera'], row['frame_idx']) == (s['episode_id'], s['camera'], s['frame_idx'])
        mask = decode(row['mask'])
        assert mask.shape == (s['image_height'], s['image_width'])
        assert row['visibility'] in {'visible', 'partial_occlusion', 'out_of_view', 'fully_occluded', 'unknown'}
        assert bool(mask.any()) == (row['visibility'] in {'visible', 'partial_occlusion'})
        if row.get('prediction_prefill'):
            assert row.get('proposal_provenance'), 'Assisted origin must remain available'
        rows[key] = row
        hashes[path.name] = sha(path)
        geometry[key] = bbox(mask), int(mask.sum())
    assert set(rows) == expected, 'Incomplete confirmed audit'
    latest = {}
    for row in map(json.loads, (source/'human_label_revisions.jsonl').read_text().splitlines()):
        key = row['sample_id'], row['object_id']
        assert key in expected and row['human_confirmed'] is True
        assert row['revision'] > latest.get(key, {}).get('revision', 0)
        latest[key] = row
    assert latest == rows, 'Revision history differs from current labels'
    output.mkdir(parents=True)
    for folder in ['images', 'human_labels', 'proposed_labels']:
        if (source/folder).exists():
            shutil.copytree(source/folder, output/folder)
    for name in ['manifest.json', 'human_label_revisions.jsonl']:
        shutil.copy2(source/name, output/name)
    streams = defaultdict(list)
    for s in manifest['samples']:
        for obj in s['objects']:
            key = s['sample_id'], obj['object_id']
            row = rows[key]
            box, area = geometry[key]
            streams[(s['task_id'], s['episode_id'], s['camera'])].append(dict(
                episode_id=s['episode_id'], task_id=s['task_id'], frame_idx=s['frame_idx'],
                timestamp=s['timestamp'], camera=s['camera'], sample_id=s['sample_id'],
                object_id=obj['object_id'], class_name=obj['class_name'], instance_id=row['instance_id'],
                bbox_xyxy=box, mask=row['mask'], mask_area=area,
                visible=None if row['visibility']=='unknown' else area>0,
                visibility=row['visibility'], confidence=None,
                confidence_semantics='not_available_for_reviewed_pixel_labels',
                image_width=s['image_width'], image_height=s['image_height'],
                human_confirmed=True, conditions=row.get('conditions', []),
                provenance=dict(label_path=f'human_labels/{s["sample_id"]}__{obj["object_id"]}.json',
                    label_sha256=hashes[f'{s["sample_id"]}__{obj["object_id"]}.json'],
                    source_image_sha256=s['image_sha256'], annotation_method=row['annotation_method'],
                    prediction_prefill=row.get('prediction_prefill', False),
                    proposal_provenance=row.get('proposal_provenance'), revision=row['revision'])))
    for (task, ep, camera), values in streams.items():
        path = output/'objects'/f'task_{task:02d}'/ep/f'{camera}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as f:
            for row in sorted(values, key=lambda r: (r['frame_idx'], r['object_id'])):
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    report = dict(result='PASS', source=str(source.resolve()), frames=len(samples),
        confirmed_labels=len(rows), manifest_sha256=sha(output/'manifest.json'),
        source_label_hashes=hashes, visibility=dict(Counter(r['visibility'] for r in rows.values())),
        origins=dict(Counter('assisted_user_confirmed' if r.get('prediction_prefill') else 'original_manual'
                             for r in rows.values())), nonempty=sum(area>0 for _, area in geometry.values()),
        object_streams=len(streams), robot_masks='not in human object audit; existing robot outputs unchanged',
        script_sha256=sha(__file__))
    write_json(output/'import_report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = import_audit(a.source, a.reference, a.output)
    print(json.dumps({k: v for k, v in report.items() if k!='source_label_hashes'}, indent=2))


if __name__ == '__main__':
    main()
