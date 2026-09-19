import unittest
import numpy as np
from seed_bank_recovery import new_track, action_for, check_reasons


class RecoveryTest(unittest.TestCase):
    def test_eval_frame_never_reseeded(self):
        tracks={'recovery_bank':{1:new_track()}}
        self.assertEqual(check_reasons(15,tracks,115,115),[])
        self.assertTrue(check_reasons(10,tracks,115,110))

    def test_shared_schedule_uses_either_arm(self):
        track=new_track();track['active']=True
        self.assertEqual(check_reasons(5,{'recovery_bank':{1:track}},None,5),[])
        self.assertTrue(check_reasons(5,{'recovery_bank':{1:track},'recovery_detector':{1:new_track()}},None,5))

    def test_reacquisition_not_claimed_as_first_detection(self):
        t=new_track();c={'binary':np.ones((8,8),bool)}
        self.assertEqual(action_for(t,c,[]),('seed','first_visible_candidate'))
        t['ever_seeded']=True
        self.assertEqual(action_for(t,c,[]),('seed','reacquire'))

    def test_robot_overlap_alone_keeps_supported_mask(self):
        t=new_track();t.update(active=True,last=np.ones((8,8),bool),flags=['robot_overlap_warning'])
        self.assertEqual(action_for(t,{'binary':t['last']},[])[0],'keep')
        for _ in range(4): self.assertEqual(action_for(t,None,[])[0],'keep')

    def test_termination_requires_repeated_missing_support(self):
        t=new_track();t.update(active=True,empty=4,flags=['empty_mask'])
        self.assertEqual(action_for(t,None,[])[0],'keep')
        self.assertEqual(action_for(t,None,[])[0],'keep')
        self.assertEqual(action_for(t,None,[])[0],'terminate')

    def test_supported_track_preserved_but_jump_reseeded(self):
        t=new_track();t.update(active=True,last=np.ones((8,8),bool))
        c={'binary':t['last']}
        self.assertEqual(action_for(t,c,[])[0],'keep')
        t['flags']=['mask_iou_drop']
        self.assertEqual(action_for(t,c,[])[0],'seed')


if __name__=='__main__': unittest.main()
