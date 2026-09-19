"""Validate dependency-free codec and label export against canonical COCO."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np
from pycocotools import mask as coco
from audit_rle import encode_counts,decode_counts,validate_counts
from annotate import BASE


def counts_for(a):
    counts=[];prev=0;run=0
    for pixel in a.flatten(order='F'):
        if pixel==prev:run+=1
        else:counts.append(run);run=1;prev=int(pixel)
    counts.append(run);return counts


class PortableTests(unittest.TestCase):
    def test_exact_coco_compatibility(self):
        rng=np.random.default_rng(20260916)
        for h,w in [(1,1),(3,11),(71,53),(720,1280)]:
            for probability in [0.,.02,.5,.98,1.]:
                a=(rng.random((h,w))<probability).astype(np.uint8);runs=counts_for(a)
                result=encode_counts(runs,h,w);expected=coco.encode(np.asfortranarray(a))
                self.assertEqual(result['counts'],expected['counts'].decode('ascii'))
                self.assertEqual(decode_counts(result),runs)

    def test_invalid_runs_rejected(self):
        for runs in [[],[4,1],[-1,5],[True,3],[2.0,2]]:
            with self.assertRaises(ValueError):validate_counts(runs,2,2)

    def test_export_without_site_packages(self):
        with tempfile.TemporaryDirectory(prefix='mask-audit-export-') as tmp:
            root=Path(tmp);audit=root/'audit';labels=audit/'human_labels';labels.mkdir(parents=True)
            sample=dict(sample_id='fixture',image_sha256='fixture-image',image_height=2,image_width=2,objects=[dict(object_id=2)])
            manifest=json.dumps(dict(samples=[sample])).encode();(audit/'manifest.json').write_bytes(manifest)
            row=dict(sample_id='fixture',object_id=2,source_image_sha256='fixture-image',mask=encode_counts([4],2,2),
                     human_confirmed=True,annotation_method='manual_pixel_editor',annotator='SYNTHETIC TEST',
                     instance_id='banana_1',visibility='out_of_view')
            (labels/'fixture__2.json').write_text(json.dumps(row))
            archive=root/'return.zip'
            subprocess.run([sys.executable,'-S',str(BASE/'export_audit_labels.py'),'--bundle',str(root),'--output',str(archive)],
                           check=True,stdout=subprocess.DEVNULL)
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip());info=json.loads(z.read('return_info.json'))
                self.assertEqual(info['audit_manifest_sha256'],hashlib.sha256(manifest).hexdigest())
                self.assertEqual(info['labels'],1);self.assertEqual(info['completed_frames'],1)
                self.assertEqual(json.loads(z.read('human_labels/fixture__2.json')),row)
                self.assertFalse(any(name.startswith('images/') for name in z.namelist()))
                for name,value in info['files_sha256'].items():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),value)


if __name__=='__main__':unittest.main()
