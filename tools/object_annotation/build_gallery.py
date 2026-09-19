"""Build a local HTML index of QA videos; no web service or upload."""
import argparse
from html import escape
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
body=['<!doctype html><meta charset="utf-8"><title>Astribot bbox QA</title>',
      '<style>body{font-family:system-ui;max-width:1320px;margin:2em auto;background:#15191e;color:#eee}video{width:100%;max-width:1280px}a{color:#8ed2ff}section{margin:2em 0}</style>',
      '<h1>Astribot bounding-box pilot QA</h1>',
      '<p>Left: original RGB. Right: detections. Videos use sampled frames, not interpolation. These are candidate annotations, not ground truth.</p>']
for path in sorted((a.root/'qa').rglob('*.mp4')):
    rel=path.relative_to(a.root).as_posix();label=path.relative_to(a.root/'qa').as_posix()
    body.append(f'<section><h2>{escape(label)}</h2><video controls preload="none" src="{escape(rel)}"></video><p><a href="{escape(rel)}">Open/download video</a></p></section>')
(a.root/'qa_index.html').write_text('\n'.join(body)+'\n')
print(a.root/'qa_index.html')
