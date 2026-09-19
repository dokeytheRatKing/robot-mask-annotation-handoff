"""Package a completed bounded pilot for offline viewing; no model inference."""
import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import cv2
import numpy as np

from annotate import QAVideo, sha, write_json
from assisted_seed_pilot import render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--full-metrics', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    archive = a.output.with_suffix('.zip')
    if a.output.exists() or archive.exists():
        raise FileExistsError('Use a new versioned output directory and archive')
    a.output.mkdir(parents=True)
    manifest = json.loads((a.pilot/'audit/manifest.json').read_text())
    metrics = json.loads((a.pilot/'evaluation/metrics.json').read_text())
    predictions = {}
    for path in sorted(a.pilot.glob('task_*/episode_*/*.assisted.jsonl')):
        dest = a.output/'masks'/path.relative_to(a.pilot)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        for row in map(json.loads, path.read_text().splitlines()):
            key = row['sample_id'], row['object_id']
            assert key not in predictions
            predictions[key] = row
    clips = defaultdict(list)
    for sample in manifest['samples']:
        clips[sample['clip_id']].append(sample)
    qa = a.output/'qa'
    qa.mkdir()
    sections, decoded = [], []
    for clip_id, samples in clips.items():
        frames = []
        camera = samples[0]['camera']
        video_path = qa/f'{camera}.mp4'
        writer = QAVideo(video_path, 6)
        for sample in sorted(samples, key=lambda s:s['frame_idx']):
            rgb_path = a.pilot/'audit'/sample['image']
            assert sha(rgb_path) == sample['image_sha256']
            rows = [predictions[sample['sample_id'], o['object_id']] for o in sample['objects']]
            canvas = render(cv2.imread(str(rgb_path)), rows,
                f'{camera} frame={sample["frame_idx"]} | RGB / assisted object masks | p=SAM2 presence')
            writer.add(canvas)
            frames.append(canvas)
        writer.close()
        cv2.imwrite(str(qa/f'{camera}.jpg'), frames[len(frames)//2])
        cv2.imwrite(str(qa/f'{camera}_overview.jpg'), cv2.vconcat([cv2.resize(f,(640,196)) for f in frames]))
        cap = cv2.VideoCapture(str(video_path))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape[:2] == (392,1280) and np.std(frame)>10
            count += 1
        cap.release()
        assert count == len(samples), (video_path, count)
        webm_path = qa/f'{camera}.webm'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(video_path),
            '-an','-c:v','libvpx-vp9','-threads','2','-b:v','0','-crf','32',str(webm_path)],check=True)
        decoded.append(dict(camera=camera,frames=count,sha256=sha(video_path)))
        v = metrics['results']['assisted']['per_camera'][camera]
        sections.append(f'''<section id="{camera}"><header><h2>{html.escape(camera)}</h2>
<span>IoU {v['mask_iou']:.4f} &nbsp; Dice {v['mask_dice']:.4f}</span></header>
<video controls playsinline preload="metadata" poster="qa/{camera}.jpg"><source src="qa/{camera}.mp4" type="video/mp4"><source src="qa/{camera}.webm" type="video/webm"></video>
<p><a href="qa/{camera}_overview.jpg">{len(samples)} frames</a> &middot; <a href="qa/{camera}.mp4" download>MP4</a></p></section>''')
    shutil.copytree(a.pilot/'evaluation',a.output/'metrics_sparse')
    shutil.copytree(a.full_metrics,a.output/'metrics_full420')
    for name in ['seed_manifest.json','run_config.json','run_complete.json']:
        shutil.copy2(a.pilot/name,a.output/name)
    shutil.copy2(a.report,a.output/'REPORT.md')
    fields = ['method','camera','labeled_object_frames','visible_gt','mask_iou','mask_dice',
              'visibility_accuracy','id_switch_count','id_comparisons','out_of_view_samples','out_of_view_residue_rate',
              'eligible_reentry_events','reentry_success_rate']
    with (a.output/'metrics_sparse/metrics.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for method,values in metrics['results'].items():
            for camera,values in values['per_camera'].items():
                writer.writerow(dict(method=method,camera=camera,**{k:values[k] for k in fields[2:]}))
    table = []
    for camera in ['head','left_wrist','right_wrist']:
        scores = [metrics['results'][method]['per_camera'][camera]['mask_iou']
                  for method in ['baseline','recovery','assisted']]
        table.append('<tr><th>'+camera+'</th>'+''.join(f'<td>{s:.4f}</td>' for s in scores)+'</tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Astribot mask review</title>
<style>*{box-sizing:border-box}body{font:15px system-ui;color:#202626;background:#fafafa;margin:0;letter-spacing:0}
main{max-width:1180px;margin:auto;padding:24px 16px}h1{font-size:26px;margin:0 0 8px}h2{font-size:20px;margin:0}
p{line-height:1.5;color:#505858}nav{display:flex;flex-wrap:wrap;gap:20px;margin:20px 0}a{color:#126447}
section{border-top:1px solid #cdd4d2;padding:20px 0}header{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:12px}
video{display:block;width:100%;aspect-ratio:1280/392;background:#111}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:12px 6px;border-bottom:1px solid #d6dddb}td{font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}.legend span:before{content:"";display:inline-block;width:10px;height:10px;background:var(--color);margin-right:5px}
</style><main><h1>Astribot / Task 24</h1><p>Episode 002478 &middot; 60 images &middot; 6 seed frames &middot; 54 scored frames</p>
<nav><a href="#head">Head</a><a href="#left_wrist">Left wrist</a><a href="#right_wrist">Right wrist</a><a href="REPORT.md">Report</a><a href="metrics_sparse/metrics.csv">Metrics CSV</a></nav>
<div class="legend"><span style="--color:#ffdc23">Banana</span><span style="--color:#ff8c2d">Basket</span><span style="--color:#eb5fdc">Peach</span><span style="--color:#f04150">Watermelon</span><span style="--color:#41eb7d">Avocado</span></div>
'''+''.join(sections)+'''<section><h2>Non-seed mask IoU</h2><table><thead><tr><th>Camera</th><th>Baseline</th><th>Recovery</th><th>Assisted</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table>
<p>Reviewed sparse-seed development pilot. Offline bidirectional SAM2.1 base-plus. Predictions remain assisted drafts.</p></section></main></html>'''
    (a.output/'index.html').write_text(page)
    (a.output/'README.txt').write_text('Open index.html in a local browser. No web server or model is required.\n'
        'REPORT.md contains methods, limitations, source paths and reproduction commands.\n'
        'This is a view-only package, not a request to relabel the confirmed 420 images.\n'
        'Video p is SAM2 presence, not calibrated mask accuracy. Robot masks are unchanged and not shown.\n'
        'masks/ contains unchanged prediction JSONL. Reference labels remain on the server.\n')
    write_json(a.output/'media_acceptance.json',dict(result='PASS',videos=decoded,
        script_sha256=sha(__file__),source_render_sha256=sha(Path(__file__).with_name('assisted_seed_pilot.py'))))
    files = {str(path.relative_to(a.output)):sha(path) for path in sorted(a.output.rglob('*')) if path.is_file()}
    write_json(a.output/'PACKAGE_MANIFEST.json',dict(files=files))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(a.output.rglob('*')):
            if path.is_file():
                z.write(path,str(Path(a.output.name)/path.relative_to(a.output)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(dict(result='PASS',archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha(archive),videos=decoded),indent=2))


if __name__ == '__main__':
    main()
