"""Read-only real-data browser acceptance; never click save on confirmed labels."""
import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

from annotate import PROJECT, sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8766')
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    before={path.name:sha(path) for path in (a.audit/'human_labels').glob('*.json')}
    history=sha(a.audit/'human_label_revisions.jsonl')
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(PROJECT/'.cache/playwright'))
    results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        for name,width,height in [('desktop',1440,1050),('mobile',390,844)]:
            page=browser.new_page(viewport=dict(width=width,height=height));errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            for idx in [61,143,242,420]:
                # A changed query forces a fresh document, unlike a hash-only navigation.
                page.goto(a.url+f'/?review_frame={idx}#frame={idx}')
                page.wait_for_function('typeof frameReady!=="undefined" && frameReady && !loading')
                assert page.evaluate('pos+1')==idx
                assert page.locator('#all').is_checked() and page.locator('#show').is_checked()
                assert page.evaluate('layers.size')==5
                assert page.evaluate('new Set([...layers.keys()].map(color)).size')==5
                colored=page.evaluate('v.toDataURL()')
                page.locator('#show').uncheck()
                assert page.evaluate('v.toDataURL()')!=colored, 'Mask overlay missing'
                page.locator('#show').check()
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Horizontal overflow'
                page.screenshot(path=str(a.output/f'{name}_{idx}.png'),full_page=True)
                results.append(dict(viewport=name,source_ui_index=idx,layers=5,overlay_nonblank=True))
            assert not errors,errors
            page.close()
        browser.close()
    assert before=={path.name:sha(path) for path in (a.audit/'human_labels').glob('*.json')}
    assert history==sha(a.audit/'human_label_revisions.jsonl')
    write_json(a.output/'acceptance.json',dict(result='PASS',screens=results,human_labels_unchanged=len(before)))
    print(json.dumps(dict(result='PASS',screens=len(results),human_labels_unchanged=len(before))))


if __name__=='__main__':
    main()
