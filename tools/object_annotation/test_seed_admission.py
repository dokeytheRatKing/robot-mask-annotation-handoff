import unittest
import numpy as np
from seed_admission_pilot import admission


class AdmissionTest(unittest.TestCase):
    def candidate(self, oid=1):
        return dict(object_id=oid,binary=np.ones((8,8),bool),detector_score=.6,
                    sam_predicted_iou=.9,retrieval=dict(similarity=.7,margin=.1),scores=dict(masked=.8))

    def test_requires_two_observations(self):
        self.assertEqual(admission(self.candidate(),[],None,10),(False,'await_second_observation'))

    def test_old_or_disjoint_observation_rejected(self):
        c=self.candidate();p=dict(c,frame_idx=0)
        self.assertFalse(admission(c,[],p,20)[0])
        p.update(frame_idx=5,binary=np.zeros((8,8),bool))
        self.assertEqual(admission(c,[],p,10)[1],'short_window_disagreement')

    def test_identity_conflict(self):
        c=self.candidate();p=dict(c,frame_idx=5)
        self.assertEqual(admission(c,[self.candidate(2)],p,10)[1],'cross_identity_collision')

    def test_valid_and_weak_evidence(self):
        c=self.candidate();p=dict(c,frame_idx=5)
        self.assertTrue(admission(c,[c],p,10)[0])
        c['retrieval']['margin']=.01
        self.assertFalse(admission(c,[],p,10)[0])
        c['retrieval']['margin']=.1;c['sam_predicted_iou']=.5
        self.assertEqual(admission(c,[],p,10)[1],'sam_boundary_confidence_low')

    def test_no_candidate(self):
        self.assertEqual(admission(None,[],None,10),(False,'no_candidate'))


if __name__=='__main__':unittest.main()
