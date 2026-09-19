import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

import full_queue as queue
from full_segmentation import blocks, LazyImages, candidates_for_task, episode_frames
from recovery import Settings


class FullSegmentationTests(unittest.TestCase):
    def test_preflight_cap_preserves_source_timestamp_contract(self):
        episode={'frames':360,'preflight_original_frames':1246}
        with patch('full_segmentation.frames',return_value=iter([])) as reader:
            list(episode_frames(episode,'head'))
        self.assertEqual(reader.call_args.args[0]['frames'],1246)
        self.assertEqual(reader.call_args.kwargs['max_frames'],360)
        self.assertEqual(episode['frames'],360)
    def test_container_does_not_reject_contained_object(self):
        hits=[dict(object_id=5,confidence=.8,bbox_xyxy=[0,0,100,100]),
              dict(object_id=22,confidence=.7,bbox_xyxy=[20,20,40,40])]
        chosen,rejected=candidates_for_task(hits,[5,22],Settings())
        self.assertEqual(set(chosen),{5,22});self.assertFalse(rejected)

    def test_identical_different_fruit_boxes_remain_ambiguous(self):
        hits=[dict(object_id=i,confidence=.8,bbox_xyxy=[0,0,100,100]) for i in [10,20]]
        chosen,rejected=candidates_for_task(hits,[10,20],Settings())
        self.assertFalse(chosen);self.assertEqual(set(rejected),{10,20})

    def test_overlap_has_every_frame_exactly_once(self):
        for length in [1,119,120,121,240,241,5803]:
            actual=[];last=None
            for batch,overlap in blocks(iter(range(length)),120):
                if overlap: self.assertEqual(batch[0],last)
                actual.extend(batch[1:] if overlap else batch)
                last=batch[-1]
            self.assertEqual(actual,list(range(length)))

    def test_lazy_adapter_matches_official_pixels(self):
        bgr=np.random.default_rng(42).integers(0,256,(37,63,3),dtype=np.uint8)
        lazy=LazyImages([bgr]*6,64,'cpu')
        rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).convert('RGB').resize((64,64))
        expected=torch.from_numpy(np.array(rgb)/255.).permute(2,0,1).float()
        expected=(expected-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        self.assertTrue(torch.allclose(lazy[0],expected,atol=1e-6,rtol=0))
        for i in range(6): lazy[i]
        self.assertEqual(len(lazy.cache),3)

    def test_queue_claim_and_completion_are_distinct_from_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            queue.initialize(root,[('episode_1',c,1,17,0) for c in ['head','left_wrist','right_wrist']])
            first=queue.claim(root,0);second=queue.claim(root,1)
            self.assertNotEqual(first['id'],second['id'])
            queue.heartbeat(root,first,12)
            queue.finish(root,first,dict(wall_seconds=1,frames=17,review_events=7))
            queue.fail(root,second,'recoverable')
            repeated=queue.claim(root,1)
            self.assertEqual(repeated['id'],second['id'])
            self.assertEqual(repeated['attempts'],2)
            status=queue.status(root)
            self.assertEqual(status['completed_episodes'],0)
            self.assertEqual(status['recent_review_events'],7)
            self.assertEqual(status['recent_frames'],17)


if __name__=='__main__': unittest.main()
