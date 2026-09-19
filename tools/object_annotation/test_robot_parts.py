import unittest
import numpy as np

from robot_parts import flange_split, resolve_conflicts, anatomy_guard, PARTS
from full_segmentation import candidates_for_task
from recovery import Settings


class RobotPartsTests(unittest.TestCase):
    def test_unobserved_flange_falls_back_to_robot_not_false_ee(self):
        mask=np.ones((5,5),bool)
        guarded,reasons=anatomy_guard({1103:mask},'head',set(),mask.shape)
        self.assertFalse(guarded[1103].any());self.assertTrue(guarded[1100].all())
        self.assertIn(1103,reasons)
        guarded,reasons=anatomy_guard({1103:mask},'left_wrist',set(),mask.shape)
        self.assertTrue(guarded[1103].all());self.assertFalse(reasons)
        guarded,reasons=anatomy_guard({1103:mask},'head',{'left'},mask.shape)
        self.assertTrue(guarded[1103].all());self.assertFalse(reasons)
    def test_flange_uncertainty_is_not_invented_anatomy(self):
        mask=np.ones((25,25),bool)
        arm,ee,unknown=flange_split(mask,[[0,12],[24,12]],[12,24],band=2)
        self.assertTrue(arm[20,12]);self.assertTrue(ee[2,12]);self.assertTrue(unknown[12,12])
        self.assertTrue(np.array_equal(arm|ee|unknown,mask))
        self.assertFalse((arm&ee).any())

    def test_overlap_becomes_unknown_without_touching_object(self):
        arm=np.zeros((10,10),bool);arm[1:6,2:8]=True
        ee=np.zeros_like(arm);ee[4:9,2:8]=True
        masks={1101:arm,1103:ee};object_mask=ee.copy()
        parts=resolve_conflicts(masks,arm.shape)
        self.assertEqual(set(parts),set(PARTS))
        self.assertTrue(np.array_equal(parts[1100],arm&ee))
        self.assertTrue(np.array_equal(np.logical_or.reduce(list(parts.values())),arm|ee))
        self.assertTrue(np.array_equal(ee,object_mask))
        self.assertTrue((np.sum(list(parts.values()),axis=0)<=1).all())

    def test_appearance_veto_overrides_large_grounding_confidence(self):
        wrong=dict(object_id=1,confidence=.99,bbox_xyxy=[0,0,10,10],appearance_accepted=False)
        right=dict(object_id=1,confidence=.55,bbox_xyxy=[20,20,40,40],appearance_accepted=True)
        chosen,_=candidates_for_task([wrong,right],[1],Settings())
        self.assertEqual(chosen[1],right)


if __name__=='__main__':unittest.main()
