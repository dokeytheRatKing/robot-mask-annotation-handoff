import unittest
import numpy as np
from masks import encode,decode,bbox,record


class MaskTests(unittest.TestCase):
    def test_noncontiguous_roundtrip(self):
        a=np.zeros((11,17),np.uint8);a[0:3,0:7]=1;a[10,16]=1
        for value in [a,a[:,::-1],a.T]:
            self.assertTrue(np.array_equal(value,decode(encode(value))))
        self.assertEqual(bbox(a),[0,0,17,11])

    def test_empty_explicit(self):
        r=record(np.zeros((9,13),bool),confidence=.9)
        self.assertFalse(r['visible']);self.assertIsNone(r['bbox_xyxy'])
        self.assertEqual(r['mask_status'],'predicted_empty')
        self.assertEqual(int(decode(r['mask']).sum()),0)

    def test_overlap_retained(self):
        obj=np.zeros((10,10),bool);obj[2:6,2:6]=True
        robot=np.zeros_like(obj);robot[4:9,4:9]=True
        self.assertEqual(int((decode(encode(obj))&decode(encode(robot))).sum()),4)
        self.assertEqual(bbox(obj),[2,2,6,6])


if __name__=='__main__':unittest.main()
