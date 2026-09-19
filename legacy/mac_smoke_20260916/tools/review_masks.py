"""Render proposal contact sheets and per-frame overlays for visual review."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from assist_masks import AUDIT, WORK, decode, human_rows

COLORS = {1: (45, 210, 255), 2: (255, 220, 35), 4: (255, 140, 45),
          10: (235, 95, 220), 20: (240, 65, 80), 21: (65, 235, 125)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clips', nargs='+', type=int)
    p.add_argument('--individual', action='store_true')
    p.add_argument('--final', action='store_true', help='Render the actual human + pending labels')
    args = p.parse_args()
    manifest = json.loads((AUDIT / 'manifest.json').read_text())
    humans = human_rows()
    proposals = {(r['sample_id'], r['object_id']): r
                 for path in (AUDIT / 'proposed_labels').glob('*.json')
                 for r in [json.loads(path.read_text())]} if args.final else {}
    out = WORK / ('final_overlays' if args.final else 'overlays')
    out.mkdir(exist_ok=True)
    for ci, clip in enumerate(manifest['clips'], 1):
        if args.clips and ci not in args.clips:
            continue
        path = WORK / 'predictions' / (clip['clip_id'] + '.npz')
        if not path.exists() and not args.final:
            continue
        pred = np.load(path) if not args.final else None
        packed = pred['masks'] if pred is not None else None
        ss = [(i+1, s) for i, s in enumerate(manifest['samples']) if s['clip_id'] == clip['clip_id']]
        sheet = Image.new('RGB', (1600, 1240), '#19232c')
        d = ImageDraw.Draw(sheet)
        for j, (idx, s) in enumerate(ss):
            arr = np.array(Image.open(AUDIT / s['image']).convert('RGB'))
            ids = []
            objects = [o['object_id'] for o in s['objects']] if args.final else pred['object_ids']
            for k, oid in enumerate(objects):
                if args.final:
                    row = humans.get((s['sample_id'], int(oid))) or proposals[s['sample_id'], int(oid)]
                    mask = decode(row['mask'])
                else:
                    mask = packed[j, k]
                if (s['sample_id'], int(oid)) in humans:
                    mask = decode(humans[s['sample_id'], int(oid)]['mask'])
                if mask.any():
                    arr[mask] = (.5 * arr[mask] + .5 * np.array(COLORS[int(oid)])).astype(np.uint8)
                    ids.append(int(oid))
            im = Image.fromarray(arr)
            if args.individual:
                im.save(out / f'{idx:03d}.jpg', quality=96)
            x, y = j%4*400, j//4*248
            sheet.paste(im.resize((400, 225)), (x, y+23))
            d.text((x+4, y+4), f'UI {idx} | {ids}', fill='white')
        sheet.save(out / f'{ci:02d}_{clip["clip_id"]}.jpg', quality=93)
        print('rendered', ci)


if __name__ == '__main__':
    main()
