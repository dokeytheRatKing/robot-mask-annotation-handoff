"""Check frozen inputs, admission evidence, causal provenance and dense output."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from annotate import BASE, sha, write_json
from masks import bbox, decode
from seed_bank import SeedBank
from seed_bank_transfer import WEIGHTS, choose
from seed_admission_pilot import DEFAULT, MODES


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.


def static_accept(candidate, pool):
    if candidate is None:
        return False
    r = candidate['retrieval']
    if r['similarity'] is None or r['margin'] is None:
        return False
    if r['similarity'] < .50 or r['margin'] < .05 or candidate['sam_predicted_iou'] < .80:
        return False
    return not any(
        c['object_id'] != candidate['object_id'] and c['detector_score'] >= .30
        and iou(c['binary'], candidate['binary']) > .70
        and c['scores']['masked'] >= candidate['scores']['masked'] - .10
        for c in pool
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--seed-revision', default='seeds_r2')
    a = p.parse_args()
    cfg = json.loads((a.root / 'run_config.json').read_text())
    parent = Path(cfg['parent'])
    assert sha(parent / 'run_config.json') == cfg['parent_config_sha256']
    assert sha(Path(cfg['bank']) / 'bank.json') == cfg['bank_sha256']
    for name, digest in cfg['code'].items():
        assert sha(BASE / name) == digest
    bank = SeedBank(Path(cfg['bank']))
    exemplars = {r['exemplar_id']: r for r in bank.entries}
    seed_dir = a.root / a.seed_revision
    seed_manifest = json.loads((seed_dir / 'manifest.json').read_text())
    review = json.loads((seed_dir / 'review.json').read_text())
    assert seed_manifest['model_sha256'] == cfg['models']['sam2_sha256'] == sha(WEIGHTS)
    assert seed_manifest['config_sha256'] == sha(a.root / 'run_config.json')
    assert review['decision'] == 'use_as_assisted_seeds' and review['human_confirmed'] is False
    assert review['seed_sha256'] == seed_manifest['seed_sha256'] == sha(seed_dir / 'seeds.json')
    seeds = json.loads((seed_dir / 'seeds.json').read_text())
    assert [r['prompt'] for r in seeds] == seed_manifest['prompts']
    prompt_keys = lambda rows: [(r['case_id'], r['frame_idx'], r['object_id']) for r in rows]
    assert prompt_keys(seeds) == prompt_keys(cfg['prompts'])
    for r in seeds:
        assert r['human_confirmed'] is False
        assert sha(seed_dir / f'{r["case_id"]}_{r["frame_idx"]}_rgb.jpg') == r['rgb_sha256']
        mask = decode(r['mask'])
        assert bbox(mask) == r['bbox_xyxy'] and int(mask.sum()) == r['mask_area']
        assert bool(mask.any()) == (r['prompt']['status'] == 'visible')
    counts = Counter()
    unknown = Counter()
    for case in cfg['cases']:
        cid = case['case_id']; output = a.root / cid; source = parent / cid
        meta = json.loads((output / 'complete.json').read_text())
        prior = json.loads((source / 'complete.json').read_text())
        assert meta['case'] == case and meta['config_sha256'] == sha(a.root / 'run_config.json')
        assert meta['seed_sha256'] == review['seed_sha256']
        assert meta['review_sha256'] == sha(seed_dir / 'review.json')
        assert meta['events_sha256'] == sha(output / 'events.json')
        assert prior['candidates_sha256'] == sha(source / 'candidates.jsonl')
        assert prior['checks_sha256'] == sha(source / 'checks.json')
        assert sha(prior['robot_source']['path']) == prior['robot_source']['sha256']
        reference_path = source / 'single_bank.jsonl.gz'
        assert sha(reference_path) == prior['output_sha256']['single_bank']
        with gzip.open(reference_path, 'rt') as f:
            reference = {(r['frame_idx'], r['object_id']): r for r in map(json.loads, f)}
        checks = sorted(r['frame_idx'] for r in json.loads((source / 'checks.json').read_text()))
        assert len(checks) == len(set(checks)) and case['eval_frame'] not in checks
        assert checks[0] == case['start']
        pool = defaultdict(list); byid = {}
        for line in (source / 'candidates.jsonl').read_text().splitlines():
            r = json.loads(line); r['binary'] = decode(r['mask'])
            assert r['candidate_id'] not in byid
            byid[r['candidate_id']] = r; pool[r['frame_idx']].append(r)
            assert r['frame_idx'] in checks and r['object_id'] in case['object_ids']
            assert bbox(r['binary']) == r['bbox_xyxy'] and int(r['binary'].sum()) == r['mask_area']
            refs = r['retrieval']['nearest_exemplar_ids']
            if r['retrieval'].get('competitor_exemplar_id'):
                refs = refs + [r['retrieval']['competitor_exemplar_id']]
            for eid in refs:
                assert exemplars[eid]['bank_status'] == 'active'
                assert exemplars[eid]['episode_id'] != case['episode']['episode_id']
                counts['episode_excluded_references'] += 1
        chosen = {}; admitted = {}; previous = {oid: None for oid in case['object_ids']}
        for idx in checks:
            for oid in case['object_ids']:
                selected, _ = choose([r for r in pool[idx] if r['object_id'] == oid], 'bank_rank', True)
                chosen[idx, oid] = selected
                good = static_accept(selected, pool[idx]); prev = previous[oid]
                admitted[idx, oid] = bool(good and prev and idx-prev['frame_idx'] <= 15 and iou(selected['binary'], prev['binary']) >= .30)
                previous[oid] = selected if good else None
        proposals = {(r['frame_idx'], r['object_id']): r for r in seeds if r['case_id'] == cid}
        for idx, oid in proposals:
            assert idx != case['eval_frame'] and case['start'] <= idx <= case['end']
        events = json.loads((output / 'events.json').read_text())
        indexed = {(e['mode'], e['frame_idx'], e['object_id']): e for e in events}
        assert len(indexed) == len(events)
        expected_events = {(m, idx, oid) for m in MODES for idx in checks for oid in case['object_ids']}
        expected_events |= {('assisted', idx, oid) for idx, oid in proposals}
        assert set(indexed) == expected_events
        for mode in MODES:
            path = output / f'{mode}.jsonl.gz'; assert sha(path) == meta['output_sha256'][mode]
            active = {oid: None for oid in case['object_ids']}
            verified = dict(active); last_mask = dict(active); seen = set(); times = {}
            with gzip.open(path, 'rt') as f:
                for r in map(json.loads, f):
                    idx = r['frame_idx']; oid = r['object_id']; key = (idx, oid)
                    assert key not in seen; seen.add(key)
                    e = indexed.get((mode, idx, oid))
                    if e:
                        if mode == 'assisted':
                            seed = proposals.get(key)
                            action = ('seed' if seed['prompt']['status'] == 'visible' else 'clear_unknown') if seed else 'keep'
                            assert e['action'] == action and e['candidate_id'] is None
                        else:
                            c = chosen[key]
                            assert e['candidate_id'] == (c['candidate_id'] if c else None)
                            allowed = c is not None if mode == 'guarded_replace' and idx == case['start'] else admitted[key]
                            action = 'keep'
                            if allowed:
                                verified[oid] = idx
                                if active[oid] is None or iou(last_mask[oid], c['binary']) < .25:
                                    action = 'seed'
                            elif mode == 'guarded_all' and active[oid] is not None and idx-verified[oid] >= 30:
                                action = 'clear_unknown'
                            assert e['action'] == action, (cid, mode, idx, oid, e, action)
                        if action in ('seed', 'clear_unknown'):
                            assert idx != case['eval_frame']
                            active[oid] = e if action == 'seed' else None
                        counts[f'{mode}/{action}/{e["reason"]}'] += 1
                    assert r['mode'] == mode and r['human_confirmed'] is False
                    assert r['episode_id'] == case['episode']['episode_id'] and r['camera'] == case['camera']
                    for field in ('episode_id', 'task_id', 'camera', 'timestamp', 'class_name', 'image_width', 'image_height'):
                        assert r[field] == reference[key][field]
                    assert np.isfinite(r['timestamp']); times[idx] = r['timestamp']
                    if r['mask'] is None:
                        assert active[oid] is None and r['visible'] is None and r['visibility'] == 'unknown'
                        assert r['bbox_xyxy'] is None and r['confidence'] is None and r['seed_frame'] is None
                        unknown[mode] += 1; last_mask[oid] = None
                    else:
                        assert active[oid] is not None and r['seed_frame'] == active[oid]['frame_idx'] <= idx
                        assert r['seed_frame'] != case['eval_frame']
                        provenance = r['provenance']
                        assert provenance['robot_subtraction'] is False and provenance['human_confirmed'] is False
                        if mode == 'assisted':
                            assert provenance['seed_sha256'] == review['seed_sha256']
                            assert provenance['prompt'] == proposals[r['seed_frame'], oid]['prompt']
                        else:
                            assert provenance['candidate_id'] == active[oid]['candidate_id']
                            assert provenance['retrieval'] == byid[provenance['candidate_id']]['retrieval']
                            assert provenance['parent_candidate_sha256'] == prior['candidates_sha256']
                        mask = decode(r['mask']); last_mask[oid] = mask
                        assert mask.shape == (r['image_height'], r['image_width'])
                        assert bbox(mask) == r['bbox_xyxy'] and int(mask.sum()) == r['mask_area']
                        assert bool(mask.any()) == r['visible']
                        assert np.isfinite(r['confidence']) and 0 <= r['confidence'] <= 1
                    counts['rows'] += 1
            assert seen == {(idx, oid) for idx in range(case['start'], case['end']+1) for oid in case['object_ids']}
            assert np.all(np.diff([times[idx] for idx in sorted(times)]) >= 0)
        counts['cases'] += 1
        print(f'Checked {cid}', flush=True)
    result = dict(status='PASS', counts=dict(counts), unknown_rows=dict(unknown),
                  config_sha256=sha(a.root / 'run_config.json'), validator_sha256=sha(Path(__file__)),
                  seed_sha256=review['seed_sha256'], scope='Geometry, evidence, query exclusion, provenance and causality, NOT semantic acceptance')
    write_json(a.root / 'independent_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
