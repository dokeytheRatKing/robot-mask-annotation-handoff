"""Synthetic I/O checks for version-dependent HDF5 and identity-safe seed binding."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import h5py
import numpy as np
from PIL import Image

from common import load_clip, load_mask_seeds, resolve_under, save, sha
from extract_hdf5_clip import decode_rgb
from propagate_clip import intervals

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


class PortabilityTests(unittest.TestCase):
    def test_decoded_raw_and_encoded_channel_order(self):
        rgb = np.zeros((6, 9, 3), np.uint8); rgb[:, :, 0] = 255
        np.testing.assert_array_equal(decode_rgb(rgb, 'RGB'), rgb)
        np.testing.assert_array_equal(decode_rgb(rgb[:, :, ::-1], 'BGR'), rgb)
        ok, encoded = cv2.imencode('.png', rgb[:, :, ::-1]); self.assertTrue(ok)
        np.testing.assert_array_equal(decode_rgb(encoded.tobytes(), 'BGR'), rgb)
        with self.assertRaises(ValueError): decode_rgb(rgb.astype('float32'), 'RGB')

    def test_hdf5_inventory_extract_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'source').mkdir(); src = root/'source/episode.hdf5'
            array = np.zeros((4, 6, 9, 3), np.uint8)
            for i in range(4): array[i, :, :, 0] = i*50
            with h5py.File(src, 'w') as f:
                f['observation/head_camera/rgb'] = array
                f['timestamp'] = [100., 100.2, 100.2, 100.6]
                f['actor_segmentation'] = np.zeros((4, 6, 9), np.uint32)
            before = sha(src)
            inventory = root/'inventory.json'
            subprocess.run([sys.executable, str(BASE/'inspect_robotwin.py'), '--input', str(src),
                '--output', str(inventory)], check=True, capture_output=True)
            items = json.loads(inventory.read_text())['files'][0]['datasets']
            self.assertTrue(any(r['segmentation_candidate'] for r in items))
            for timed in (True, False):
                dest = root/('timed' if timed else 'untimed')
                cmd = [sys.executable, str(BASE/'extract_hdf5_clip.py'), '--input', str(src),
                    '--rgb-key', 'observation/head_camera/rgb', '--channel-order', 'RGB',
                    '--task-id', 'task_demo', '--episode-id', 'episode_demo', '--camera', 'head_camera',
                    '--start', '1', '--count', '2', '--output', str(dest)]
                if timed: cmd += ['--timestamp-key', 'timestamp']
                subprocess.run(cmd, check=True, capture_output=True)
                _, data = load_clip(dest/'clip.json')
                self.assertEqual([r['source_frame_idx'] for r in data['frames']], [1, 2])
                self.assertEqual([r['timestamp'] for r in data['frames']], [100.2,100.2] if timed else [None,None])
                np.testing.assert_array_equal(np.asarray(Image.open(dest/'rgb/00000.png')), array[1])
            self.assertEqual(sha(src), before)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): resolve_under(tmp, '../other.png')
            with self.assertRaises(ValueError): resolve_under(tmp, '/tmp/other.png')

    def test_exact_seed_image_binding(self):
        _, clip = load_clip(ROOT/'examples/kettle_wrist/clip.json')
        path, seeds = load_mask_seeds(ROOT/'examples/kettle_wrist/seeds.json', clip)
        with tempfile.TemporaryDirectory() as tmp:
            changed = json.loads(path.read_text()); changed['seeds'][0]['source_image_sha256'] = '0'*64
            save(Path(tmp)/'seeds.json', changed)
            with self.assertRaises(AssertionError): load_mask_seeds(Path(tmp)/'seeds.json', clip)

    def test_nearest_seed_intervals_cover_once(self):
        seeds = [dict(object_id=1,local_frame_idx=2), dict(object_id=1,local_frame_idx=7)]
        parts = list(intervals(seeds, 10))
        indices = [j for _,_,left,right in parts for j in range(left,right+1)]
        self.assertEqual(indices, list(range(10)))


if __name__ == '__main__':
    unittest.main()
