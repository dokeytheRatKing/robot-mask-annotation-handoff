"""Validate the download, then open its static review page at two viewport sizes."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

from playwright.sync_api import sync_playwright
from PIL import Image

from annotate import PROJECT, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(PROJECT/'.cache/playwright'))
    with tempfile.TemporaryDirectory(prefix='astribot-review-check-') as temp:
        root = Path(temp)
        with zipfile.ZipFile(a.archive) as z:
            assert z.testzip() is None
            assert len(z.namelist()) == len(set(z.namelist()))
            for name in z.namelist():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
            z.extractall(root)
        package = root/a.archive.stem
        manifest = json.loads((package/'PACKAGE_MANIFEST.json').read_text())
        for name,expected in manifest['files'].items():
            assert hashlib.sha256((package/name).read_bytes()).hexdigest() == expected, name
        screens = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True,args=['--no-sandbox'])
            for name,width,height in [('desktop',1440,1000),('mobile',390,844)]:
                page = browser.new_page(viewport=dict(width=width,height=height))
                errors = []
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto((package/'index.html').as_uri())
                page.wait_for_function('Array.from(document.querySelectorAll("video")).every(v=>v.readyState>=2 && v.videoWidth>0)')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                for camera in ['head','left_wrist','right_wrist']:
                    video = page.locator(f'#{camera} video')
                    video.scroll_into_view_if_needed()
                    video.evaluate('(v)=>{v.muted=true;return v.play()}')
                    page.wait_for_function(f'document.querySelector("#{camera} video").currentTime>0.2')
                    video.evaluate('(v)=>v.pause()')
                    # Browser file:// media may taint canvas; inspect rendered pixels instead.
                    pixels = Image.open(io.BytesIO(video.screenshot())).convert('RGB').resize((64,32))
                    assert len(set(pixels.getdata())) > 32, camera
                page.screenshot(path=str(a.output/f'{name}.png'),full_page=True)
                assert not errors, errors
                screens.append(dict(viewport=name,video_count=3,played=True,nonblank=True,no_horizontal_overflow=True))
                page.close()
            browser.close()
    result = dict(result='PASS',files_verified=len(manifest['files']),screens=screens)
    write_json(a.output/'acceptance.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
