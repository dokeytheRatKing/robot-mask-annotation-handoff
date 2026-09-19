"""Integration checks: proposals stay unconfirmed until an explicit human save."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
import zipfile

BUNDLE=Path(__file__).resolve().parents[1]/'astribot_mask_audit_420_20260916'
sys.path.insert(0,str(BUNDLE))
from audit_rle import decode_counts
from export_labels import export


class ReviewIntegration(unittest.TestCase):
    def test_proposal_review_and_export(self):
        with tempfile.TemporaryDirectory(prefix='astribot-review-test-') as tmp:
            bundle=Path(tmp);root=bundle/'audit';root.mkdir()
            manifest=json.loads((BUNDLE/'audit/manifest.json').read_text())
            sample=manifest['samples'][0];manifest['samples']=[sample]
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'images').symlink_to(BUNDLE/'audit/images',target_is_directory=True)
            (root/'proposed_labels').mkdir()
            original=json.loads((BUNDLE/'audit/human_labels'/f'{sample["sample_id"]}__2.json').read_text())
            proposal={**original,'human_confirmed':False,'annotation_method':'sam2_1_video_assisted',
                      'prediction_prefill':True,'proposal_id':'test-proposal','model':'sam2.1_hiera_small'}
            name=f'{sample["sample_id"]}__2.json'
            raw=json.dumps(proposal).encode();(root/'proposed_labels'/name).write_bytes(raw)
            process=subprocess.Popen([sys.executable,'-B',str(BUNDLE/'audit_server.py'),str(root),'--port','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                url=process.stdout.readline().split()[-1]
                def get(route):
                    with urllib.request.urlopen(url+route,timeout=5) as r:return json.load(r)
                def post(body):
                    req=urllib.request.Request(url+'/save',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Origin':url})
                    with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)
                self.assertEqual(get('/status')['proposals'],1)
                self.assertEqual(get('/status')['labels'],0)
                rows=get('/labels?id='+sample['sample_id'])
                self.assertFalse(rows[0]['human_confirmed'])
                self.assertEqual(rows[0]['label_source'],'proposal')
                with self.assertRaises(ValueError):export(bundle,bundle/'must_not_export.zip')
                body={**proposal,'mask_rle_counts':decode_counts(proposal['mask'])}
                with self.assertRaises(urllib.error.HTTPError):post(body)
                self.assertFalse(list((root/'human_labels').glob('*.json')))
                body.update(human_confirmed=True,annotator='ISOLATED_TEST_REVIEWER',notes='reviewed in isolated test')
                self.assertEqual(post(body)['revision'],1)
                saved=get('/label?id='+sample['sample_id']+'&object=2')
                self.assertTrue(saved['human_confirmed']);self.assertTrue(saved['prediction_prefill'])
                self.assertEqual(saved['proposal_provenance']['proposal_id'],'test-proposal')
                self.assertEqual(saved['label_source'],'human')
                self.assertEqual(get('/status')['proposals'],0)
                self.assertEqual(get('/status')['labels'],1)
                self.assertEqual((root/'proposed_labels'/name).read_bytes(),raw)
                body['notes']='corrected review';self.assertEqual(post(body)['revision'],2)
                self.assertEqual(len((root/'human_label_revisions.jsonl').read_text().splitlines()),2)
                archive=export(bundle,bundle/'confirmed_only.zip')
                with zipfile.ZipFile(archive) as z:
                    self.assertFalse(any(n.startswith('proposed_labels') for n in z.namelist()))
                    info=json.loads(z.read('return_info.json'));self.assertEqual(info['labels'],1)
                    self.assertEqual(info['completed_frames'],0)
                    for n,h in info['files_sha256'].items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),h)
            finally:
                process.terminate();process.communicate(timeout=5)


if __name__=='__main__':unittest.main()
