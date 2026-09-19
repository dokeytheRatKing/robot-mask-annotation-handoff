import unittest
import numpy as np
from recovery import Settings,choose_candidates,diagnostics,terminate_reason


class RecoveryTests(unittest.TestCase):
    def test_robot_contact_is_warning_not_termination(self):
        m=np.zeros((100,100),bool);m[20:40,20:40]=True
        flags=diagnostics({1:m},{1:m},m,Settings())[1]
        self.assertEqual(flags,['robot_overlap_warning'])
        self.assertIsNone(terminate_reason(flags,0,20,.1,Settings()))
        self.assertEqual(m.sum(),400)

    def test_missing_detection_alone_does_not_erase_object(self):
        self.assertIsNone(terminate_reason([],0,20,.9,Settings()))
        self.assertIsNone(terminate_reason(['tiny_border_residue'],0,2,.9,Settings()))
        self.assertEqual(terminate_reason(['tiny_border_residue'],0,3,.9,Settings()),'suspected_out_of_view_residue')

    def test_cross_class_overlap_abstains(self):
        ds=[dict(object_id=i,confidence=.8,bbox_xyxy=[1,1,20,20]) for i in [1,2]]
        candidates,rejected=choose_candidates(ds,[1,2],Settings())
        self.assertEqual(candidates,{})
        self.assertEqual(set(rejected),{1,2})

    def test_displacement_triggers_redetection(self):
        a=np.zeros((100,100),bool);b=a.copy();a[5:15,5:15]=True;b[70:80,70:80]=True
        flags=diagnostics({1:b},{1:a},None,Settings())[1]
        self.assertIn('mask_iou_drop',flags);self.assertIn('bbox_center_jump',flags)


if __name__=='__main__':unittest.main()
