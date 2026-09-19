import unittest
import numpy as np

from seed_bank import compatible, crops, diverse_indices, quality_reason, rank_score, retrieve


class SeedBankTests(unittest.TestCase):
    def test_aliases_are_explicit(self):
        self.assertFalse(compatible(dict(object_id=17),22))
        self.assertTrue(compatible(dict(object_id=17,merged_object_ids=[17,22,23]),22))

    def test_episode_exclusion_across_cameras(self):
        entries = [dict(object_id=1,episode_id='a',camera='head',exemplar_id='a'),
                   dict(object_id=1,episode_id='a',camera='left_wrist',exemplar_id='b')]
        f = np.eye(2)
        r = retrieve(entries,f,f,f[0],f[0],1,'head',exclude_episode='a')
        self.assertEqual(r['status'],'no_reference')
        self.assertIsNone(r['similarity'])

    def test_camera_preference_and_competitor(self):
        entries = [dict(object_id=1,episode_id='a',camera='head',exemplar_id='a'),
                   dict(object_id=1,episode_id='b',camera='left_wrist',exemplar_id='b'),
                   dict(object_id=1103,episode_id='c',camera='head',exemplar_id='c')]
        f = np.array([[.8,.6],[1,0],[.9,.4358899]])
        r = retrieve(entries,f,f,np.array([1,0]),np.array([1,0]),1,'head')
        self.assertEqual(r['nearest_exemplar_ids'],['a'])
        self.assertEqual(r['competitor_id'],1103)
        self.assertAlmostEqual(r['margin'],-.1)

    def test_cross_camera_fallback(self):
        e = [dict(object_id=1,episode_id='a',camera='head',exemplar_id='a')]
        f = np.array([[1.,0.]])
        r = retrieve(e,f,f,f[0],f[0],1,'right_wrist')
        self.assertEqual(r['camera_scope'],'cross_camera_fallback')
        self.assertIsNone(r['margin'])

    def test_crop_mask_coordinates(self):
        im = np.full((20,30,3),255,np.uint8); mask = np.zeros((20,30),bool); mask[5:15,8:20]=True
        rgb,masked,m,box = crops(im,mask)
        self.assertEqual(int(m.sum()),120)
        self.assertTrue((masked[~m]==127).all())
        self.assertTrue((masked[m]==255).all())
        self.assertEqual(rgb.shape[:2],m.shape)
        self.assertEqual(int(mask[box[1]:box[3],box[0]:box[2]].sum()),120)

    def test_empty_not_exemplar(self):
        self.assertEqual(quality_reason(dict(visible=False),np.zeros((5,5),bool)),'not_visible')
        with self.assertRaises(ValueError):
            crops(np.zeros((5,5,3),np.uint8),np.zeros((5,5),bool))

    def test_diversity_excludes_adjacent_copies(self):
        rows = [dict(episode_id='a',frame_idx=i,mask_area=1000) for i in (0,1,60)]
        chosen = diverse_indices(rows,np.eye(3),6)
        self.assertEqual(chosen,[0,2])

    def test_no_reference_fallback(self):
        self.assertEqual(rank_score(.6,dict(similarity=None)),.6)

    def test_rejected_reference_not_used(self):
        e = [dict(object_id=1,episode_id='a',camera='head',exemplar_id='a',bank_status='rejected')]
        f = np.array([[1.,0.]])
        self.assertEqual(retrieve(e,f,f,f[0],f[0],1,'head')['status'],'no_reference')


if __name__ == '__main__':
    unittest.main()
