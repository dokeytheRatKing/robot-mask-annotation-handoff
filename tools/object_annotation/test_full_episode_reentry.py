import unittest
from full_episode_reentry import temporal_requests, merge_requests


class ReviewQueueTest(unittest.TestCase):
    def test_unknown_is_reviewed_and_deduplicated(self):
        rows={i:[dict(object_id=2,mask=None,suspicious_flags=[])] for i in range(200)}
        req=temporal_requests(rows,[2])
        self.assertEqual([r['frame_idx'] for r in req if r['reason']=='unseeded_or_cleared'],[0,90,180])

    def test_distinct_identity_and_failure_preserved(self):
        rows={1:[dict(object_id=2,mask={},suspicious_flags=['mask_iou_drop']),
                 dict(object_id=4,mask={},suspicious_flags=['robot_overlap_warning'])]}
        req=merge_requests(temporal_requests(rows,[2,4]),100)
        self.assertEqual(req[0]['object_ids'],[2,4])
        self.assertEqual(len(req[0]['triggers']),2)
        self.assertEqual(req[0]['context_frames'],[0,1,16])
        self.assertEqual(req[0]['status'],'pending_assistant_review')

    def test_never_silently_marks_review_complete(self):
        req=merge_requests([dict(frame_idx=99,object_id=1,reason='empty_mask')],100)
        self.assertEqual(req[0]['context_frames'],[84,99])
        self.assertEqual(req[0]['status'],'pending_assistant_review')

    def test_periodic_check_catches_stable_wrong_tracks(self):
        rows={i:[dict(object_id=1,mask={},suspicious_flags=[])] for i in range(241)}
        self.assertEqual([r['frame_idx'] for r in temporal_requests(rows,[1])],[0,120,240])


if __name__=='__main__':unittest.main()
