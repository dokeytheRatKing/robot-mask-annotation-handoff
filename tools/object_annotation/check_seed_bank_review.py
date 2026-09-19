"""Exercise portable review UI without touching actual accepted labels."""
import json
import os
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright
from annotate import PROJECT, write_json


def main():
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(PROJECT/'.cache/playwright'))
    bank=PROJECT/'annotations/identity_seed_bank_20260918_curated'
    with tempfile.TemporaryDirectory(prefix='seed-review-test-') as temp, sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1440,height=1000));errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((bank/'review.html').as_uri())
        assert page.locator('article').count()==265
        page.locator('#status').select_option('retrieval_excluded')
        assert page.locator('article').count()==24
        page.locator('#accept').click()
        with page.expect_download() as info:page.locator('#export').click()
        path=Path(temp)/'review.json';info.value.save_as(path)
        data=json.loads(path.read_text());assert len(data['decisions'])==24
        assert all(d['decision']=='accept' for d in data['decisions'].values())
        page.reload();page.locator('#status').select_option('retrieval_excluded')
        assert page.locator('article select').first.input_value()=='accept'
        page.locator('#import').set_input_files(str(path))
        page.wait_for_function("document.getElementById('notice').textContent === 'Review imported'")
        invalid=Path(temp)/'wrong_bank.json'
        data['bank_sha256']='wrong';write_json(invalid,data)
        page.locator('#import').set_input_files(str(invalid))
        page.wait_for_function("document.getElementById('notice').textContent === 'Bank hash mismatch'")
        page.locator('#status').select_option('active')
        page.locator('#identity').select_option('21');page.locator('#camera').select_option('head')
        assert page.locator('article').count()>0
        # Force all visible image assets to decode, including lazy elements.
        page.evaluate("Promise.all(Array.from(document.images).map(im=>{im.loading='eager';return im.decode()}))")
        assert page.evaluate('Array.from(document.images).every(im=>im.naturalWidth>0)')
        out=PROJECT/'annotations/identity_seed_bank_pilot_20260918_curated'
        page.screenshot(path=str(out/'review_desktop.png'))
        page.set_viewport_size(dict(width=390,height=844))
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out/'review_mobile.png'))
        assert not errors,errors
        browser.close()
        write_json(out/'review_browser_test.json',dict(status='PASS',entries=265,excluded=24,
            checks=['file:// without server','filters','bulk acceptance','export','persistence',
                    'import','wrong bank rejected','image decoding','desktop/mobile layout'],page_errors=errors))
        print('PASS portable seed review')


if __name__=='__main__':main()
