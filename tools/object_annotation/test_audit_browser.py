"""Real browser test against disposable synthetic fixtures, never the real GT."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from annotate import BASE,PROJECT,write_json,sha
from masks import decode,encode


def main():
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(PROJECT/'.cache/playwright'))
    with tempfile.TemporaryDirectory(prefix='astribot-audit-ui-') as temp:
        root=Path(temp);Image.new('RGB',(64,48),(80,90,100)).save(root/'fixture.png')
        sample=dict(sample_id='synthetic_fixture',episode_id='fixture',camera='head',frame_idx=0,clip_id='fixture',
            image='fixture.png',image_width=64,image_height=48,image_sha256=sha(root/'fixture.png'),objects=[dict(object_id=2,class_name='banana'),dict(object_id=4,class_name='basket')])
        write_json(root/'manifest.json',dict(samples=[sample],required_conditions=['grasp_contact']))
        (root/'proposed_labels').mkdir()
        other=np.zeros((48,64),bool);other[1:8,1:8]=True
        write_json(root/'proposed_labels/synthetic_fixture__4.json',dict(sample_id='synthetic_fixture',object_id=4,
            visibility='visible',instance_id='basket_1',conditions=[],mask=encode(other),
            human_confirmed=False,prediction_prefill=True,annotation_method='test_proposal',proposal_id='fixture-basket'))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        proc=subprocess.Popen([sys.executable,'-S',str(BASE/'audit_server.py'),str(root),'--port',str(port)],stdout=subprocess.DEVNULL)
        try:
            for _ in range(80):
                try:urlopen(f'http://127.0.0.1:{port}/manifest',timeout=1);break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('Test server failed to start')
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
                page=browser.new_page(viewport=dict(width=1400,height=1100));errors=[]
                page.on('pageerror',lambda exc:errors.append(str(exc)))
                page.goto(f'http://127.0.0.1:{port}')
                page.wait_for_function("document.getElementById('view').width === 64 && !loading")
                assert page.locator('#all').is_checked()
                other_pixels=page.evaluate('layers.get(4).toDataURL()')
                combined=page.evaluate('v.toDataURL()')
                page.locator('#all').uncheck()
                assert page.evaluate('v.toDataURL()')!=combined
                page.locator('#all').check()
                page.locator('#annotator').fill('SYNTHETIC TEST ONLY')
                page.locator('#visibility').select_option('partial_occlusion')
                rect=page.locator('#view').bounding_box()
                for x,y in [(16,12),(48,12),(48,36),(16,36)]:
                    page.mouse.click(rect['x']+x/64*rect['width'],rect['y']+y/48*rect['height'])
                page.locator('#polygon').click();page.locator('#confirm').check();page.locator('#save').click()
                page.wait_for_function("document.getElementById('notice').textContent.includes('已保存 revision 1')")
                saved=json.loads((root/'human_labels/synthetic_fixture__2.json').read_text())
                binary=decode(saved['mask']);assert 700<int(binary.sum())<850
                assert page.evaluate('layers.get(4).toDataURL()')==other_pixels
                page.reload();page.wait_for_function("!loading && document.getElementById('notice').textContent.includes('revision 1')")
                restored=page.evaluate('Array.from(m.getImageData(0,0,mc.width,mc.height).data).filter((v,i)=>i%4===3).map(v=>v>127)')
                assert np.array_equal(np.array(restored).reshape(48,64),binary),'Browser COCO RLE decoder differs from pycocotools'
                page.locator('#clear').click();page.locator('#confirm').check();page.locator('#save').click()
                page.wait_for_function("document.getElementById('notice').textContent.includes('mask') && !saving")
                assert json.loads((root/'human_labels/synthetic_fixture__2.json').read_text())['revision']==1
                page.locator('#visibility').select_option('out_of_view');page.locator('#confirm').check();page.locator('#save').click()
                page.wait_for_function("document.getElementById('notice').textContent.includes('已保存 revision 2')")
                saved=json.loads((root/'human_labels/synthetic_fixture__2.json').read_text());assert not decode(saved['mask']).any()
                assert len((root/'human_label_revisions.jsonl').read_text().splitlines())==2
                page.evaluate('window.savedReadJSON=readJSON;readJSON=async()=>{throw Error("TEST_LOAD_FAILURE")};load()')
                page.wait_for_function('!loading && !frameReady')
                assert page.locator('#save').is_disabled()
                assert not page.locator('#jump').is_disabled()
                assert page.evaluate('layers.size')==0
                page.evaluate('readJSON=window.savedReadJSON;load()')
                page.wait_for_function('!loading && frameReady')
                page.evaluate('selectObject(4)')
                page.locator('#confirm').check();page.locator('#save').click()
                page.wait_for_function("document.getElementById('notice').textContent.includes('已保存 revision 1') && !saving")
                reviewed=json.loads((root/'human_labels/synthetic_fixture__4.json').read_text())
                assert reviewed['prediction_prefill'] and reviewed['proposal_provenance']['proposal_id']=='fixture-basket'
                assert not errors,errors
                browser.close()
            print('PASS: polygon/RLE persistence, multi-mask default and layer isolation, proposal provenance, visibility gates, failed-load save protection; synthetic fixture only.')
        finally:
            proc.terminate();proc.wait(timeout=10)


if __name__=='__main__':main()
