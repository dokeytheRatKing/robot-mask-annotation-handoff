"""Check the shipped ZIP after relocation, with site packages disabled in server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import zipfile
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);a=p.parse_args()
    project=Path(__file__).resolve().parents[2]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(project/'.cache/playwright'))
    with tempfile.TemporaryDirectory(prefix='portable-mask-bundle-') as temp:
        with zipfile.ZipFile(a.archive) as z:
            assert z.testzip() is None
            for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
            z.extractall(temp)
        roots=list(Path(temp).iterdir());assert len(roots)==1;root=roots[0]
        for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
            expected,name=line.split('  ',1)
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        proc=subprocess.Popen([sys.executable,'-S',str(root/'audit_server.py'),'--port',str(port)],
                              cwd=temp,stdout=subprocess.DEVNULL)
        try:
            for _ in range(80):
                try:status=json.load(urlopen(f'http://127.0.0.1:{port}/status',timeout=1));break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('Relocated server did not start')
            assert status['frames']==420 and status['required_labels']==1860 and status['labels']==0
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
                page=browser.new_page(viewport=dict(width=1440,height=1050));errors=[]
                page.on('pageerror',lambda exc:errors.append(str(exc)))
                page.goto(f'http://127.0.0.1:{port}')
                page.wait_for_function('typeof manifest !== "undefined" && manifest.samples.length === 420 && !loading')
                page.locator('#index').fill('420');page.locator('#jump').click()
                page.wait_for_function('pos === 419 && !loading')
                assert page.evaluate('img.naturalWidth')>0
                assert not errors,errors
                browser.close()
            assert not list((root/'audit/human_labels').glob('*.json'))
            print('PASS: ZIP CRC, all packaged SHA256 files, 420 frames, relocated stdlib-only server, browser first/last frame, no GT writes.')
        finally:proc.terminate();proc.wait(timeout=10)


if __name__=='__main__':main()
