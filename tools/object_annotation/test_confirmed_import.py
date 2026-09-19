import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from annotate import sha, write_json
from audit_server import current_label
from import_confirmed_audit import import_audit
from masks import encode, decode


class ConfirmedImportTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'; (source/'images').mkdir(parents=True); (source/'human_labels').mkdir()
        (source/'images/test.png').write_bytes(b'hashed fixture; no image decode needed by importer')
        sample = dict(sample_id='test',episode_id='episode_1',task_id=1,camera='head',frame_idx=0,
            timestamp=1.,image='images/test.png',image_sha256=sha(source/'images/test.png'),
            image_height=3,image_width=4,objects=[dict(object_id=1,class_name='kettle')])
        row = dict(sample_id='test',episode_id='episode_1',camera='head',frame_idx=0,object_id=1,
            source_image_sha256=sample['image_sha256'],mask=encode(np.ones((3,4),bool)),
            visibility='partial_occlusion',instance_id='kettle_1',human_confirmed=True,annotator='FIXTURE',
            annotation_method='manual_pixel_editor',prediction_prefill=True,
            proposal_provenance=dict(proposal_id='fixture'),revision=1)
        write_json(source/'manifest.json',dict(samples=[sample]))
        write_json(source/'human_labels/test__1.json',row)
        (source/'human_label_revisions.jsonl').write_text(json.dumps(row)+'\n')
        return source,row

    def test_import_preserves_bytes_and_assisted_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,row=self.fixture(root); output=root/'imported'
            result=import_audit(source,source,output)
            self.assertEqual(result['origins'],dict(assisted_user_confirmed=1))
            self.assertEqual((source/'human_labels/test__1.json').read_bytes(),(output/'human_labels/test__1.json').read_bytes())
            exported=json.loads((output/'objects/task_01/episode_1/head.jsonl').read_text())
            self.assertIsNone(exported['confidence'])
            self.assertEqual(exported['bbox_xyxy'],[0,0,4,3])
            self.assertTrue(np.array_equal(decode(exported['mask']),decode(row['mask'])))
            with self.assertRaises(AssertionError):import_audit(source,source,output)

    def test_inconsistent_history_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,row=self.fixture(root)
            (source/'human_label_revisions.jsonl').write_text(json.dumps(row|dict(revision=2))+'\n')
            with self.assertRaises(AssertionError):import_audit(source,source,root/'imported')
            self.assertFalse((root/'imported').exists())

    def test_confirmed_precedence_over_historical_proposal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,row=self.fixture(root)
            (source/'proposed_labels').mkdir()
            write_json(source/'proposed_labels/test__1.json',row|dict(human_confirmed=False))
            chosen=current_label(source,'test',1)
            self.assertTrue(chosen['human_confirmed']); self.assertEqual(chosen['label_source'],'human')
            self.assertNotIn('label_source',json.loads((source/'human_labels/test__1.json').read_text()))


if __name__=='__main__':
    unittest.main()
