import unittest
import numpy as np
from pathlib import Path
import tempfile
import cv2

from masks import record
from seed_bank_transfer import choose, panel
from seed_bank_transfer_qa import Video


def proposal(cid,det,score,similarity=.7,margin=.1):
    return dict(candidate_id=cid,detector_score=det,scores=dict(detector=det,masked=score),
                retrieval=dict(similarity=similarity,margin=margin))


class TransferSelectionTest(unittest.TestCase):
    def test_ranking_uses_same_pool(self):
        candidates=[proposal('a',.9,.4),proposal('b',.6,.8)]
        self.assertEqual(choose(candidates,'detector')[0]['candidate_id'],'a')
        self.assertEqual(choose(candidates,'bank_rank')[0]['candidate_id'],'b')

    def test_gate_abstains_not_fallback_to_wrong_identity(self):
        candidates=[proposal('a',.9,.9,margin=-.1),proposal('b',.6,.8)]
        self.assertEqual(choose(candidates,'bank_gate'),(None,'low_identity_margin'))
        self.assertIsNotNone(choose(candidates,'bank_rank')[0])

    def test_missing_reference_not_positive(self):
        self.assertEqual(choose([proposal('a',.8,.8,None,None)],'bank_gate'),(None,'no_reference'))

    def test_empty_and_low_detector(self):
        for mode in ('detector','bank_rank','bank_gate'):
            self.assertEqual(choose([],mode),(None,'no_candidate'))
            self.assertEqual(choose([proposal('a',.29,.9)],mode),(None,'detector_below_threshold'))

    def test_eligibility_first_does_not_discard_eligible_second_choice(self):
        pool=[proposal('low',.29,.9),proposal('eligible',.7,.8)]
        self.assertIsNone(choose(pool,'bank_rank')[0])
        self.assertEqual(choose(pool,'bank_rank',True)[0]['candidate_id'],'eligible')

    def test_ties_deterministic(self):
        c=[proposal('b',.8,.8),proposal('a',.8,.8)]
        for mode in ('detector','bank_rank','bank_gate'):
            self.assertEqual(choose(c,mode)[0]['candidate_id'],'a')

    def test_legacy_robot_id_drawing_does_not_mutate_record(self):
        row=dict(object_id='robot',class_name='robot',confidence=.8,
                 **record(np.ones((16,16),dtype=bool)))
        out=panel(np.zeros((16,16,3),dtype=np.uint8),[row],'Robot QA')
        self.assertEqual(out.shape,(266,426,3))
        self.assertEqual(row['object_id'],'robot')

    def test_qa_video_uses_actual_geometry_and_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.mp4';writer=Video(path,10)
            writer.add(np.full((32,48,3),30,dtype=np.uint8))
            writer.add(np.full((32,48,3),220,dtype=np.uint8))
            writer.close();cap=cv2.VideoCapture(str(path))
            ok,first=cap.read();self.assertTrue(ok)
            ok,last=cap.read();self.assertTrue(ok)
            self.assertEqual(first.shape,(32,48,3))
            self.assertGreater(last.mean()-first.mean(),150)
            self.assertFalse(cap.read()[0]);cap.release()


if __name__=='__main__':unittest.main()
