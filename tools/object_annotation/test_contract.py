"""Targeted checks for timestamp/frame identities, missing data, and jump flags."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from data import scan,select,frames
from diagnostics import compare,geometry


class ContractTests(unittest.TestCase):
    def test_timestamp_adapter_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);camera=root/'task_01'/'episode_002'
            for name in ['head','left_wrist','right_wrist']:
                (camera/name).mkdir(parents=True)
                for i in [0,1,2,3,10]:
                    cv2.imwrite(str(camera/name/f'{i}.png'),np.full((12,16,3),i,dtype=np.uint8))
            eps=scan(root);self.assertEqual(len(eps),1)
            self.assertEqual(eps[0]['task_id'],1)
            e=select(eps,[1],1,1)[0]
            # Timestamp-free folders must yield null, not synthetic acquisition times.
            fs=list(frames(e,'head',2));self.assertEqual([x[0] for x in fs],[0,2,4])
            self.assertTrue(all(x[1] is None for x in fs))
            self.assertEqual(int(fs[-1][2][0,0,0]),10)
            e['cameras']['head']['timestamps']=[100,101,102,103,110]
            self.assertEqual([x[1] for x in frames(e,'head',2)],[100,102,110])

    def test_missing_does_not_make_boxes(self):
        old=[dict(object_id=1,bbox_xyxy=[0,0,10,10],confidence=.9)]
        current=[];r=compare(old,current,5,0,[1],100,100)
        self.assertEqual(current,[])
        self.assertEqual(r[0]['reasons'],['detection_disappeared'])
        self.assertEqual(r[0]['frame_gap'],5)
        self.assertNotIn('bbox_xyxy',r[0])

    def test_geometry_and_suspicion(self):
        self.assertEqual(geometry([0,0,10,10],[0,0,10,10],100,100)['iou'],1)
        a=[dict(object_id=1,bbox_xyxy=[0,0,10,10],confidence=.9)]
        b=[dict(object_id=1,bbox_xyxy=[60,60,90,90],confidence=.8)]
        r=compare(a,b,10,5,[1],100,100)[0]
        self.assertEqual(set(r['reasons']),{'low_iou','center_jump','area_jump'})
        self.assertEqual(len(b),1) # flags never delete detections

    def test_document_mapping(self):
        base=Path(__file__).parent
        objects=json.loads((base/'config/objects.json').read_text())
        mapping=json.loads((base/'config/task_objects.json').read_text())
        self.assertEqual({x['object_id'] for x in objects},set(range(24)))
        self.assertEqual(set(mapping),{str(i) for i in range(30)})
        self.assertEqual(mapping['1'],[1]);self.assertEqual(mapping['9'],[1,9])
        self.assertEqual(mapping['24'],[2,4,10,20,21])
        for o in objects:
            self.assertIn(o.get('grounding_phrase',o['class_name']).lower(),o['detailed_text_prompt'].lower())


if __name__=='__main__':unittest.main()
