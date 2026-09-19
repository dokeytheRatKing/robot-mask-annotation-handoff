"""Run the editor CDP integration test in a disposable local audit and Chrome."""
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'astribot_mask_audit_420_20260916'
WORK = ROOT / 'work/segmentation'


def main():
    chrome_path = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    with tempfile.TemporaryDirectory(prefix='astribot-review-browser-') as tmp:
        base = Path(tmp)
        audit = base / 'audit'
        audit.mkdir()
        (audit / 'human_labels').mkdir()
        (audit / 'proposed_labels').mkdir()
        (audit / 'images').symlink_to(BUNDLE / 'audit/images', target_is_directory=True)
        manifest = json.loads((BUNDLE / 'audit/manifest.json').read_text())
        manifest['samples'] = manifest['samples'][:3]
        (audit / 'manifest.json').write_text(json.dumps(manifest))
        seed = manifest['samples'][0]
        for i, sample in enumerate(manifest['samples']):
            for obj in sample['objects']:
                row = json.loads((BUNDLE / 'audit/human_labels' / f'{seed["sample_id"]}__{obj["object_id"]}.json').read_text())
                row.update(sample_id=sample['sample_id'], frame_idx=sample['frame_idx'],
                           source_image_sha256=sample['image_sha256'])
                if i:
                    row.update(human_confirmed=False, prediction_prefill=True, proposal_id='test-proposal',
                               annotation_method='sam2_1_video_assisted', model='sam2.1_hiera_small')
                folder = audit / ('proposed_labels' if i else 'human_labels')
                (folder / f'{sample["sample_id"]}__{obj["object_id"]}.json').write_text(json.dumps(row))
        server = subprocess.Popen([sys.executable, '-B', str(BUNDLE / 'audit_server.py'), str(audit), '--port', '0'], stdout=subprocess.PIPE, text=True)
        chrome = None
        try:
            url = server.stdout.readline().split()[-1]
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            chrome = subprocess.Popen([chrome_path, '--headless=new', '--disable-gpu', '--no-first-run',
                '--no-default-browser-check', f'--user-data-dir={base / "chrome"}',
                f'--remote-debugging-port={port}', '--window-size=1450,1400', url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(100):
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=1) as r:
                        targets = json.load(r)
                    if any(t.get('url', '').startswith(url) for t in targets):
                        break
                except OSError:
                    pass
                time.sleep(.1)
            config = dict(temp=tmp, url=url, devtools_port=port, server_pid=server.pid, chrome_pid=chrome.pid)
            (WORK / 'browser_test_config.json').write_text(json.dumps(config))
            subprocess.run(['node', str(ROOT / 'tools/test_review_ui.mjs')], cwd=ROOT, check=True)
        finally:
            for process in [chrome, server]:
                if process is not None:
                    process.terminate(); process.wait(timeout=10)


if __name__ == '__main__':
    main()
