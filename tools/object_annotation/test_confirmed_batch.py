import unittest

import numpy as np

from evaluate_confirmed_batch import aggregate, prediction_mask, score
from masks import encode


class ConfirmedBatchTests(unittest.TestCase):
    def setUp(self):
        self.empty = np.zeros((4, 5), bool)
        self.one = self.empty.copy(); self.one[1, 2] = True

    def test_visible_miss_is_zero_not_excluded(self):
        result = aggregate([score(self.empty, self.one)])
        self.assertEqual(result['mean_iou_visible'], 0)
        self.assertEqual(result['misses'], 1)

    def test_empty_matches_do_not_inflate_mask_quality(self):
        result = aggregate([score(self.empty, self.empty)])
        self.assertIsNone(result['mean_iou_visible'])
        self.assertEqual(result['visibility_accuracy'], 1)

    def test_false_positive_enters_visibility_and_micro_iou(self):
        result = aggregate([score(self.one, self.empty), score(self.one, self.one)])
        self.assertEqual(result['mean_iou_visible'], 1)
        self.assertEqual(result['micro_iou'], .5)
        self.assertEqual(result['visibility_accuracy'], .5)
        self.assertEqual(result['false_positives'], 1)

    def test_meat_ids_follow_per_sample_merge(self):
        label = dict(group='objects', object_id=17, merged_object_ids=[17, 22, 23],
                     image_height=4, image_width=5)
        pred = dict(objects=[dict(object_id=i, mask=encode(self.one if i == 22 else self.empty))
                             for i in (17, 22, 23)])
        self.assertTrue(np.array_equal(prediction_mask(label, pred), self.one))
        label['merged_object_ids'] = [17]
        self.assertFalse(prediction_mask(label, pred).any())

    def test_robot_unknown_is_excluded(self):
        label = dict(group='robot', object_id=1000, image_height=4, image_width=5)
        pred = dict(robot_parts=[dict(part_id=i, mask=encode(self.one if i == 1100 else self.empty))
                                 for i in range(1100, 1105)])
        self.assertFalse(prediction_mask(label, pred).any())

    def test_missing_row_is_error_not_absence(self):
        label = dict(group='objects', object_id=1, image_height=4, image_width=5)
        with self.assertRaises(KeyError):
            prediction_mask(label, dict(objects=[]))


if __name__ == '__main__':
    unittest.main()
