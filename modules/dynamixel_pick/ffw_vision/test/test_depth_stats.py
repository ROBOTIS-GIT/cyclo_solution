from types import SimpleNamespace
import unittest

from ffw_vision.depth_stats import (
    clip_roi, color_from_msg, depth_from_msg, roi_summary, valid_mask)
import numpy as np


def make_msg(encoding, width, height, step, data, is_bigendian=0):
    return SimpleNamespace(encoding=encoding, width=width, height=height, step=step,
                           data=bytes(data), is_bigendian=is_bigendian)


class TestConversion(unittest.TestCase):

    def test_depth_with_row_padding(self):
        rows = np.array([[1.0, 2.0, -1.0], [3.0, 4.0, -1.0]], dtype='<f4')  # last col = pad
        msg = make_msg('32FC1', 2, 2, 12, rows.tobytes())
        np.testing.assert_array_equal(depth_from_msg(msg), [[1.0, 2.0], [3.0, 4.0]])

    def test_depth_16uc1_scaled_and_zero_invalid(self):
        raw = np.array([[0, 1000], [250, 65535]], dtype='<u2')
        depth = depth_from_msg(make_msg('16UC1', 2, 2, 4, raw.tobytes()), depth_scale=0.001)
        self.assertTrue(np.isnan(depth[0, 0]))
        np.testing.assert_allclose(depth[0, 1:], [1.0])
        np.testing.assert_allclose(depth[1], [0.25, 65.535], rtol=1e-6)

    def test_depth_rejects_other_encoding(self):
        with self.assertRaises(ValueError):
            depth_from_msg(make_msg('8UC1', 1, 1, 1, b'\x00'))

    def test_color_bgra_and_rgb(self):
        bgra = make_msg('bgra8', 1, 1, 4, [10, 20, 30, 255])
        np.testing.assert_array_equal(color_from_msg(bgra)[0, 0], [10, 20, 30])
        rgb = make_msg('rgb8', 1, 1, 3, [30, 20, 10])
        np.testing.assert_array_equal(color_from_msg(rgb)[0, 0], [10, 20, 30])


class TestRoi(unittest.TestCase):

    def test_zero_roi_is_full_image(self):
        self.assertEqual(clip_roi((0, 0, 0, 0), 672, 376), (0, 0, 672, 376))

    def test_roi_is_clipped(self):
        self.assertEqual(clip_roi((-5, 10, 700, 400), 672, 376), (0, 10, 672, 376))

    def test_empty_roi_raises(self):
        with self.assertRaises(ValueError):
            clip_roi((100, 100, 100, 200), 672, 376)


class TestSummary(unittest.TestCase):

    def test_valid_mask(self):
        depth = np.array([np.nan, np.inf, -np.inf, 0.05, 1.0, 20.0], dtype=np.float32)
        np.testing.assert_array_equal(
            valid_mask(depth, 0.1, 10.0), [False, False, False, False, True, False])

    def test_roi_summary(self):
        frame_a = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
        frame_b = np.array([[1.0, 1.0], [np.inf, 3.0]], dtype=np.float32)
        summary = roi_summary([frame_a, frame_b], (0, 0, 2, 2), 0.1, 10.0)
        self.assertEqual((summary['frames'], summary['pixels']), (2, 4))
        self.assertAlmostEqual(summary['valid_ratio'], 6 / 8)
        self.assertAlmostEqual(summary['valid_ratio_min'], 0.75)
        self.assertAlmostEqual(summary['median_m'], 1.5)
        self.assertAlmostEqual(summary['median_std_m'], 0.5)

    def test_roi_summary_without_valid_pixels(self):
        frame = np.full((2, 2), np.nan, dtype=np.float32)
        summary = roi_summary([frame], (0, 0, 2, 2), 0.1, 10.0)
        self.assertEqual(summary['valid_ratio'], 0.0)
        self.assertTrue(np.isnan(summary['median_m']))


if __name__ == '__main__':
    unittest.main()
