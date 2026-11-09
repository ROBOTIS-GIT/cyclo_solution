import math
import unittest

import cv2
from ffw_vision.blob_detector import normalize_angle
from ffw_vision.obb_detector import box_frame, detect_obb, ObbParams, windowed_peak
from ffw_vision.seg_detector import project
import numpy as np
from test_seg_detector import D405, K, Scene

PARAMS = ObbParams()


def top_corners(scene, length, width, height):
    """Image corners of the scene box's top face."""
    across = np.cross(scene.n, scene.axis)
    pts = [scene.center + sa * length / 2 * scene.axis + sb * width / 2 * across
           for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return project(pts, K, scene.d)


class TestDetectObb(unittest.TestCase):

    def check(self, tilt, deg, d, refine=0.0):
        scene = Scene(tilt_deg=tilt, box=(0.01, 0.0, 0.046, 0.034, deg, 0.034), d=d)
        scene.color = cv2.GaussianBlur(scene.color, (3, 3), 0)
        boxes = [(top_corners(scene, 0.046, 0.034, 0.034), 0.9)]
        params = ObbParams(refine_deg=refine)
        (obj,), _ = detect_obb(scene.color, scene.depth, K, scene.d, boxes, params)
        (cu, cv), (tu, tv) = project([scene.center, scene.center + 0.02 * scene.axis], K, scene.d)
        expected = normalize_angle(math.atan2(tv - cv, tu - cu))
        # the box itself is exact; edge votes carry about the 3 deg of the edge angle tests
        tolerance = math.radians(3 if refine else 1)
        self.assertAlmostEqual(normalize_angle(obj.angle - expected), 0.0, delta=tolerance)
        self.assertAlmostEqual(obj.plane_height, 0.034, delta=0.002)
        self.assertAlmostEqual(obj.length_m, 0.046, delta=0.002)
        self.assertAlmostEqual(obj.width_m, 0.034, delta=0.002)
        u, v = project([scene.center], K, scene.d)[0]
        self.assertAlmostEqual(obj.u, u, delta=1.0)
        self.assertAlmostEqual(obj.v, v, delta=1.0)
        self.assertEqual(obj.pose_label, 'lying')
        self.assertEqual(obj.angle_period, math.pi)
        return obj

    def test_box_straight_down(self):
        self.assertEqual(self.check(0, 30, [0.0] * 5).angle_source, 'obb')

    def test_box_tilted_view_with_distortion(self):
        for deg in (10, 65, 120):
            with self.subTest(deg=deg):
                self.check(40, deg, D405)

    def test_refine_keeps_a_good_box(self):
        self.assertEqual(self.check(30, 37, D405, refine=6.0).angle_source, 'obb+edges')

    def test_two_boxes_on_one_object_keep_the_best(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 20, 0.034))
        good = top_corners(scene, 0.046, 0.034, 0.034)
        middle = good.mean(axis=0)
        turned = middle + (good - middle) @ np.array([[0.7, 0.7], [-0.7, 0.7]])  # ~45 deg
        boxes = [(turned, 0.76), (good, 0.8)]
        (obj,), _ = detect_obb(scene.color, scene.depth, K, scene.d, boxes, PARAMS)
        self.assertEqual(obj.score, 0.8)
        scene.add_box(0.06, 0.0, 0.046, 0.034, 60, 0.034)  # a second object 6 cm away
        other = top_corners(scene, 0.046, 0.034, 0.034)
        objects, _ = detect_obb(scene.color, scene.depth, K, scene.d, [(good, 0.8), (other, 0.7)],
                                PARAMS)
        self.assertEqual(len(objects), 2)

    def test_low_score_dropped(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 0, 0.034))
        boxes = [(top_corners(scene, 0.046, 0.034, 0.034), 0.2)]
        objects, plane = detect_obb(scene.color, scene.depth, K, scene.d, boxes, PARAMS)
        self.assertEqual(objects, [])
        self.assertIsNotNone(plane)


class TestHelpers(unittest.TestCase):

    def test_box_frame_long_side_first(self):
        n = np.array([0.0, 0.0, -1.0])
        pts = np.array([[0, 0, 1], [0.01, 0, 1], [0.01, 0.03, 1], [0, 0.03, 1]], float)
        center, axis, long_side, short_side = box_frame(pts, n)
        self.assertAlmostEqual(abs(axis[1]), 1.0)
        self.assertAlmostEqual(long_side, 0.03)
        self.assertAlmostEqual(short_side, 0.01)
        np.testing.assert_allclose(center, [0.005, 0.015, 1])

    def test_windowed_peak_ignores_far_peaks(self):
        deg = np.array([10.0] * 50 + [40.0] * 100)
        self.assertAlmostEqual(windowed_peak(deg, np.ones(150), 12.0, 6.0), 10.5)
        self.assertAlmostEqual(windowed_peak(deg, np.ones(150), 38.0, 6.0), 40.5)


if __name__ == '__main__':
    unittest.main()
