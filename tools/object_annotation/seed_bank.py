"""Versioned identity exemplars and frozen DINOv2 retrieval; no training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from annotate import BASE, PROJECT, sha, write_json
from masks import bbox, decode, encode

CAMERAS = ('head', 'left_wrist', 'right_wrist')
PARTS = {1100: 'robot_unknown', 1101: 'left_arm', 1102: 'right_arm',
         1103: 'left_gripper', 1104: 'right_gripper'}
OLD = PROJECT/'annotations/mask_audit_confirmed_20260917'
ACCEPTED = PROJECT/'annotations/mask_audit_batch01_confirmed_20260918'
SNAPSHOT = PROJECT/'annotations/assistant_review_20260917_batch01'


def frame_key(row):
    return row['episode_id'], row['camera'], row['frame_idx']


def compatible(entry, object_id):
    # Only explicit per-sample aliases can merge identities.
    return object_id in (entry.get('merged_object_ids') or [entry['object_id']])


def accepted_rows():
    result = []
    for path in sorted((OLD/'objects').rglob('*.jsonl')):
        digest = sha(path)
        for row in map(json.loads, path.read_text().splitlines()):
            result.append(dict(row, group='objects', source_set='accepted420',
                rgb_path=str(OLD/'images'/f'{row["sample_id"]}.png'),
                source_manifest=str(path), source_manifest_sha256=digest))
    path = ACCEPTED/'accepted_masks.jsonl'; digest = sha(path)
    for row in map(json.loads, path.read_text().splitlines()):
        if row['group'] == 'robot':
            continue
        result.append(dict(row, source_set='accepted87',
            rgb_path=str(SNAPSHOT/row['sample_id']/'rgb.png'),
            source_manifest=str(path), source_manifest_sha256=digest))
    seen = set()
    for row in result:
        key = (*frame_key(row), row['object_id'])
        if key in seen:
            raise ValueError(f'Duplicate accepted identity/frame: {key}')
        seen.add(key)
        if not row['human_confirmed']:
            raise ValueError('Unconfirmed label in accepted source')
    return result


def quality_reason(row, mask):
    if row.get('exclude_from_metrics') or row.get('exclude_from_training'):
        return 'ignored_reference'
    if not row['visible'] or not mask.any():
        return 'not_visible'
    if row['object_id'] == 1000:
        return 'derived_union_not_identity'
    box = bbox(mask)
    if mask.sum() < 128 or min(box[2]-box[0], box[3]-box[1]) < 8:
        return 'too_small_for_identity_embedding'
    return None


def crops(image, mask):
    box = bbox(mask)
    if box is None:
        raise ValueError('Cannot embed an empty mask')
    h, w = mask.shape
    pad = max(2, round(.04*max(box[2]-box[0], box[3]-box[1])))
    x0, y0, x1, y1 = max(0, box[0]-pad), max(0, box[1]-pad), min(w, box[2]+pad), min(h, box[3]+pad)
    rgb = image[y0:y1, x0:x1].copy()
    binary = mask[y0:y1, x0:x1]
    masked = rgb.copy(); masked[~binary] = 127
    return rgb, masked, binary, [x0, y0, x1, y1]


def read_image(row, checked):
    path = row['rgb_path']
    if path not in checked:
        if sha(path) != row['provenance']['source_image_sha256']:
            raise ValueError(f'RGB hash mismatch: {path}')
        checked.add(path)
    image = cv2.imread(path)
    if image is None or image.shape[:2] != (row['image_height'], row['image_width']):
        raise ValueError(f'Invalid image: {path}')
    return image


def diverse_indices(rows, features, limit=6):
    """Prefer separate episodes then embedding diversity, avoiding adjacent copies."""
    selected = []
    remaining = set(range(len(rows)))
    while remaining and len(selected) < limit:
        allowed = [i for i in remaining if not any(
            rows[i]['episode_id'] == rows[j]['episode_id'] and
            abs(rows[i]['frame_idx']-rows[j]['frame_idx']) < 30 for j in selected)]
        if not allowed:
            break
        counts = Counter(rows[j]['episode_id'] for j in selected)
        def score(i):
            distance = 1-float((features[i]@features[selected].T).max()) if selected else 0
            return (-counts[rows[i]['episode_id']], distance,
                    min(rows[i]['mask_area'], 50000), -i)
        winner = max(allowed, key=score)
        selected.append(winner); remaining.remove(winner)
    return selected


def retrieve(entries, rgb_features, masked_features, query_rgb, query_masked,
             object_id, camera, exclude_episode=None, mode='hybrid', topk=3):
    """Cosine retrieval with an explicit same-camera preference, not a probability."""
    allowed = np.array([e['episode_id'] != exclude_episode and
                        e.get('bank_status', 'active') == 'active' for e in entries])
    positive = allowed & np.array([compatible(e, object_id) for e in entries])
    same = np.array([e['camera'] == camera for e in entries])
    if not positive.any():
        return dict(status='no_reference', similarity=None, margin=None,
                    nearest_exemplar_ids=[], competitor_id=None, competitor_similarity=None,
                    camera_scope='none', reference_count=0)
    pool = positive & same if (positive & same).any() else positive
    sr, sm = rgb_features@query_rgb, masked_features@query_masked
    sims = sr if mode == 'rgb' else sm if mode == 'masked' else .5*(sr+sm)
    # Apply the same camera policy to competitors; allow cross-camera fallback.
    negative = allowed & ~np.array([compatible(e, object_id) for e in entries])
    neg_pool = negative & same if (negative & same).any() else negative
    best = sorted(np.where(pool)[0], key=lambda i: (-float(sims[i]), entries[i]['exemplar_id']))[:topk]
    competitor = int(np.where(neg_pool)[0][np.argmax(sims[neg_pool])]) if neg_pool.any() else None
    sim = float(sims[best[0]])
    ns = float(sims[competitor]) if competitor is not None else None
    return dict(status='retrieved', similarity=sim, margin=sim-ns if ns is not None else None,
        nearest_exemplar_ids=[entries[i]['exemplar_id'] for i in best],
        nearest_similarities=[float(sims[i]) for i in best],
        competitor_id=entries[competitor]['object_id'] if competitor is not None else None,
        competitor_exemplar_id=entries[competitor]['exemplar_id'] if competitor is not None else None,
        competitor_similarity=ns, camera_scope='same_camera' if (positive & same).any() else 'cross_camera_fallback',
        reference_count=int(pool.sum()))


def rank_score(detector_score, retrieval):
    if retrieval['similarity'] is None:
        return float(detector_score)
    margin = retrieval['margin'] if retrieval['margin'] is not None else 0
    # Frozen pilot heuristic, deliberately not tuned to test labels.
    return .25*float(detector_score)+.75*retrieval['similarity']+.25*margin


def fit_tile(image, width=176, height=132):
    scale = min(width/image.shape[1], height/image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))))
    tile = np.full((height, width, 3), 35, np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    tile[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return tile


def build(output, limit=6):
    import torch
    from identity_guard import Appearance
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); checked = set(); filtered = Counter()
    rows, rgbs, masked = [], [], []
    sources = accepted_rows()
    for row in sources:
        binary = decode(row['mask'])
        reason = quality_reason(row, binary)
        if reason:
            filtered[reason] += 1; continue
        image = read_image(row, checked)
        rgb, cutout, _, _ = crops(image, binary)
        rows.append(row); rgbs.append(rgb); masked.append(cutout)
    model = Appearance()
    rgb_features = model.embed(rgbs).cpu().numpy()
    mask_features = model.embed(masked).cpu().numpy()
    joint = rgb_features+mask_features
    joint /= np.linalg.norm(joint, axis=1, keepdims=True)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['object_id'], row['camera']].append(i)
    chosen = []
    for key, indices in sorted(groups.items()):
        chosen.extend(indices[i] for i in diverse_indices([rows[j] for j in indices], joint[indices], limit))
    entries, tiles = [], []
    for number, i in enumerate(chosen):
        row = rows[i]; binary = decode(row['mask']); image = read_image(row, checked)
        rgb, cutout, crop_mask, crop_box = crops(image, binary)
        eid = f'SB{number:04d}'; directory = output/'exemplars'/str(row['object_id'])/row['camera']/eid
        directory.mkdir(parents=True)
        for name, value in [('rgb.png', rgb), ('masked_rgb.png', cutout),
                            ('mask.png', binary.astype('uint8')*255), ('crop_mask.png', crop_mask.astype('uint8')*255)]:
            if not cv2.imwrite(str(directory/name), value):
                raise RuntimeError('PNG export failed')
        entry = {k: row[k] for k in ('object_id', 'class_name', 'episode_id', 'frame_idx', 'timestamp',
            'camera', 'task_id', 'mask', 'bbox_xyxy', 'mask_area', 'image_width', 'image_height',
            'source_set', 'source_manifest', 'source_manifest_sha256', 'rgb_path', 'provenance')}
        entry.update(exemplar_id=eid, robot_part_id=row['object_id'] if row['object_id'] >= 1100 else None,
            identity_name=PARTS.get(row['object_id'], row['class_name']),
            merged_object_ids=row.get('merged_object_ids'), group=row['group'],
            tags=list(row.get('conditions', [])), visibility=row['visibility'],
            tag_provenance='copied accepted conditions; no inferred pose tags',
            crop_bbox_xyxy=crop_box, human_confirmed=True, bank_status='active',
            bank_selection_status='automatic_from_user_confirmed_reference_pending_bank_review',
            embedding_index=number,
            assets={name: str((directory/name).relative_to(output)) for name in
                    ('rgb.png', 'masked_rgb.png', 'mask.png', 'crop_mask.png')},
            asset_sha256={p.name: sha(p) for p in directory.iterdir()})
        entries.append(entry)
        tile = np.concatenate([fit_tile(rgb), fit_tile(cutout)], axis=1)
        tile = np.pad(tile, ((40, 0), (0, 0), (0, 0)))
        cv2.putText(tile, f'{eid} id{row["object_id"]} {row["camera"]}', (4, 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
        cv2.putText(tile, f'{entry["identity_name"]} {row["episode_id"][-6:]}:{row["frame_idx"]}', (4, 32), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
        tiles.append(tile)
    np.savez_compressed(output/'embeddings.npz', rgb=rgb_features[chosen], masked=mask_features[chosen],
                        exemplar_ids=np.array([e['exemplar_id'] for e in entries]))
    write_json(output/'bank.json', dict(schema='astribot.identity_seed_bank.v1', entries=entries))
    (output/'qa').mkdir()
    for page in range(0, len(tiles), 24):
        items = tiles[page:page+24]
        while len(items)%4:
            items.append(np.zeros_like(items[0]))
        sheet = np.concatenate([np.concatenate(items[j:j+4], 1) for j in range(0, len(items), 4)], 0)
        cv2.imwrite(str(output/'qa'/f'bank_{page//24:02d}.jpg'), sheet)
    objects = json.loads((BASE/'config/objects.json').read_text())
    all_ids = sorted({r['object_id'] for r in objects} | set(PARTS))
    coverage = []
    for oid in all_ids:
        for cam in CAMERAS:
            found = [e for e in entries if e['object_id'] == oid and e['camera'] == cam]
            coverage.append(dict(object_id=oid, camera=cam, exemplars=len(found),
                episodes=len({e['episode_id'] for e in found}),
                status='target_met' if len(found) >= 3 else 'sparse' if found else 'missing'))
    write_json(output/'coverage.json', coverage)
    write_json(output/'build_report.json', dict(status='COMPLETE', exemplars=len(entries),
        eligible_masks=len(rows), source_labels=len(sources), excluded=dict(filtered), max_per_identity_camera=limit,
        encoder='frozen DINOv2 ViT-S/14, 384D L2-normalized; RGB and gray-background masked crop',
        encoder_sha256=sha(PROJECT/'models/dinov2/dinov2_vits14_pretrain.pth'),
        bank_sha256=sha(output/'bank.json'), embeddings_sha256=sha(output/'embeddings.npz'),
        code_sha256=sha(__file__), embedding_code_sha256=sha(BASE/'identity_guard.py'),
        seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(),
        coverage_summary=dict(Counter(x['status'] for x in coverage)),
        note='Absent references stay missing; robot_unknown is allowed but never synthesized from named parts.'))
    print((output/'build_report.json').read_text(), flush=True)


class SeedBank:
    def __init__(self, root):
        self.root = Path(root)
        report = json.loads((self.root/'build_report.json').read_text())
        for filename, key in [('bank.json', 'bank_sha256'), ('embeddings.npz', 'embeddings_sha256')]:
            if sha(self.root/filename) != report[key]:
                raise ValueError(f'Bank hash mismatch: {filename}')
        self.entries = json.loads((self.root/'bank.json').read_text())['entries']
        data = np.load(self.root/'embeddings.npz', allow_pickle=False)
        self.rgb, self.masked = data['rgb'], data['masked']
        assert list(data['exemplar_ids']) == [e['exemplar_id'] for e in self.entries]
        for matrix in (self.rgb, self.masked):
            assert matrix.shape == (len(self.entries), 384) and np.isfinite(matrix).all()
            assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4)

    def query(self, query_rgb, query_masked, object_id, camera, exclude_episode=None, mode='hybrid'):
        return retrieve(self.entries, self.rgb, self.masked, query_rgb, query_masked,
                        object_id, camera, exclude_episode, mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=6)
    args = parser.parse_args()
    if not 3 <= args.per_group <= 10:
        parser.error('--per-group must be 3..10; sparse groups are not duplicated')
    import torch
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(20260918)
    build(args.output, args.per_group)


if __name__ == '__main__':
    main()
