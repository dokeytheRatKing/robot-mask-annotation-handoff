"""Compare frozen methods on identical non-seed frames of a completed pilot."""
import argparse
import json
from pathlib import Path

from annotate import sha, write_json
from evaluate_mask_audit import evaluate
from masks import bbox, decode


def load_predictions(root, mode):
    rows = {}
    for path in sorted(root.glob(f'task_*/episode_*/*.{mode}.jsonl')):
        for row in map(json.loads, path.read_text().splitlines()):
            key = row['episode_id'], row['camera'], row['frame_idx'], row['object_id']
            assert key not in rows
            if row['mask'] is not None:
                assert bbox(decode(row['mask'])) == row['bbox_xyxy']
            rows[key] = row
    assert rows
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--frozen-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.pilot/'audit/manifest.json').read_text())
    seeds = json.loads((a.pilot/'seed_manifest.json').read_text())
    excluded = set(seeds['sample_ids'])
    scored = {**manifest, 'samples':[s for s in manifest['samples'] if s['sample_id'] not in excluded]}
    assert not excluded & {s['sample_id'] for s in scored['samples']}
    labels, hashes = [], {}
    for s in scored['samples']:
        assert sha(a.reference/s['image']) == s['image_sha256']
        for obj in s['objects']:
            path = a.reference/'human_labels'/f'{s["sample_id"]}__{obj["object_id"]}.json'
            row = json.loads(path.read_text())
            assert row['human_confirmed'] and row['source_image_sha256'] == s['image_sha256']
            labels.append(row); hashes[path.name] = sha(path)
    predictions = {mode:load_predictions(a.frozen_run,mode) for mode in ['baseline','recovery']}
    predictions['assisted'] = load_predictions(a.pilot,'assisted')
    result, details = evaluate(scored,labels,predictions)
    result['provenance'] = dict(seed_manifest_sha256=sha(a.pilot/'seed_manifest.json'),
        pilot_config_sha256=sha(a.pilot/'run_config.json'), reference_manifest_sha256=sha(a.reference/'manifest.json'),
        excluded_seed_frames=sorted(excluded), ground_truth_hashes=hashes, script_sha256=sha(__file__),
        limitation='Assisted offline developmental comparison: seed/context review uses this scene, not a blind unseen-episode test')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'metrics.json',result); write_json(a.output/'temporal_details.json',details)
    lines = ['# Sparse Reviewed-Seed Comparison', '',
        f'{len(scored["samples"])} scored frames / {len(labels)} object records; {len(excluded)} seed frames excluded for every method.', '',
        'Same SAM2.1 base-plus checkpoint; assisted method has reviewed mask/context prompts and offline bidirectional propagation.',
        'This measures an assisted annotation workflow, not equivalent unattended/causal detector accuracy.', '',
        '| Method | Camera | IoU | Dice | Visibility | OOV Residue | ID Switches |',
        '|---|---|---:|---:|---:|---:|---:|']
    for mode, values in result['results'].items():
        for camera, v in values['per_camera'].items():
            fmt = lambda x: 'N/A' if x is None else f'{x:.4f}'
            lines.append('| '+' | '.join([mode,camera]+[fmt(v[k]) for k in
                ['mask_iou','mask_dice','visibility_accuracy','out_of_view_residue_rate','id_switch_count']])+' |')
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
