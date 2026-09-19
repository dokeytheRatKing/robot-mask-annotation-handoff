import unittest
from semantic_interval_repair import bundle_requests


class BundlingTest(unittest.TestCase):
    def test_preserves_every_trigger_and_pending_status(self):
        requests=[dict(case_id='x',frame_idx=i,triggers=[dict(frame_idx=i,object_id=o,reason='unknown') for o in (2,20)]) for i in (0,30,60,90)]
        out=bundle_requests(requests)
        self.assertEqual(len(out),2)
        self.assertEqual([i for b in out for i in b['request_ids']],list(range(4)))
        self.assertEqual(sum(len(b['all_triggers']) for b in out),8)
        self.assertTrue(all(b['status']=='pending_assistant_review' for b in out))
        self.assertEqual([r for b in out for r in b['members']],requests)

    def test_never_merges_different_camera_cases(self):
        out=bundle_requests([dict(case_id=c,frame_idx=10,triggers=[]) for c in ('head','wrist')])
        self.assertEqual(len(out),2)


if __name__=='__main__':unittest.main()
