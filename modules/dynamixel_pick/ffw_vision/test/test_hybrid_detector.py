import math
import unittest

import cv2
from ffw_vision.blob_detector import normalize_angle
from ffw_vision.hybrid_detector import detect_hybrid, HybridParams, OBB, OBB_PEAK, SEG
from ffw_vision.seg_detector import project
import numpy as np
from test_seg_detector import D405, grow, K, Scene

PARAMS = HybridParams()


def turned_axis(scene, turn_deg):
    t = math.radians(turn_deg)
    return math.cos(t) * scene.axis + math.sin(t) * np.cross(scene.n, scene.axis)


def image_angle(scene, turn_deg=0.0):
    """Image direction of the box's long axis turned on the plane."""
    tip = scene.center + 0.02 * turned_axis(scene, turn_deg)
    (cu, cv), (tu, tv) = project([scene.center, tip], K, scene.d)
    return normalize_angle(math.atan2(tv - cv, tu - cu))


def box_corners(scene, turn_deg=0.0, shift=(0.0, 0.0), grow=1.0):
    """Image corners of the box's top face, turned, shifted, grown on the plane (an OBB guess)."""
    across = np.cross(scene.n, scene.axis)
    axis = turned_axis(scene, turn_deg)
    side = np.cross(scene.n, axis)
    center = scene.center + shift[0] * scene.axis + shift[1] * across
    pts = [center + sa * 0.023 * grow * axis + sb * 0.017 * grow * side
           for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return project(pts, K, scene.d)


class TestDetectHybrid(unittest.TestCase):

    def setUp(self):
        self.scene = Scene(tilt_deg=30, box=(0.01, 0.0, 0.046, 0.034, 37, 0.034), d=D405)
        self.scene.color = cv2.GaussianBlur(self.scene.color, (3, 3), 0)
        self.segments = [(grow(self.scene.box_mask, 1), 0.9)]
        self.expected = image_angle(self.scene)

    def run_hybrid(self, boxes, segments=None):
        s = self.scene
        objects, plane = detect_hybrid(s.color, s.depth, K, s.d,
                                       self.segments if segments is None else segments,
                                       boxes, PARAMS)
        self.assertIsNotNone(plane)
        return objects

    def error(self, obj, turn_deg=0.0):
        return abs(normalize_angle(obj.angle - image_angle(self.scene, turn_deg), math.pi / 2))

    def test_seg_direction_kept_near_the_box(self):
        for turn in (0, 12, -20):
            with self.subTest(turn=turn):
                (obj,) = self.run_hybrid([(box_corners(self.scene, turn), 0.9)])
                self.assertEqual(obj.angle_source, SEG)
                self.assertLess(self.error(obj), math.radians(3))
                self.assertIsNotNone(obj.mask_outline)

    def test_long_side_from_the_mask_when_its_direction_is_kept(self):
        # the box has long and short swapped (turned 90 deg): same axis pair, other long side
        (obj,) = self.run_hybrid([(box_corners(self.scene, 90), 0.9)])
        self.assertEqual(obj.angle_source, SEG)
        long_error = abs(normalize_angle(obj.angle - self.expected, math.pi))
        self.assertLess(long_error, math.radians(3))

    def test_center_from_the_box_size_from_the_mask(self):
        shifted = box_corners(self.scene, 0, (0.003, 0.0), grow=1.2)  # boxes come out too large
        (obj,) = self.run_hybrid([(shifted, 0.9)])
        u, v = np.mean(shifted, axis=0)
        self.assertAlmostEqual(obj.u, u, delta=1.0)
        self.assertAlmostEqual(obj.v, v, delta=1.0)
        # top face points of the eroded mask: about 1 px short per side, as detect_seg
        self.assertAlmostEqual(obj.length_m, 0.043, delta=0.003)
        self.assertAlmostEqual(obj.width_m, 0.031, delta=0.003)
        self.assertFalse(obj.ambiguous)

    def test_known_face_size_reported(self):
        params = HybridParams(face_long=0.0465, face_mid=0.034, face_short=0.0285,
                              report_face_size=True)
        s = self.scene  # top 34 mm: mid side down -> long x short face
        (obj,), _ = detect_hybrid(s.color, s.depth, K, s.d, self.segments,
                                  [(box_corners(s), 0.9)], params)
        self.assertAlmostEqual(obj.length_m, 0.0465)
        self.assertAlmostEqual(obj.width_m, 0.0285)
        # the pixel sizes (what the grasp planner reads) follow the known face too
        half_long = turned_axis(s, 0) * 0.0465 / 2
        tip, tail = project([s.center + half_long, s.center - half_long], K, s.d)
        self.assertAlmostEqual(obj.length, float(np.hypot(*(tip - tail))), delta=0.5)
        self.assertEqual(obj.angle_period, math.pi)  # a rectangle, not near square
        self.assertFalse(obj.ambiguous)
        big = Scene(tilt_deg=30, box=(0.01, 0.0, 0.080, 0.060, 37, 0.034), d=D405)  # two in one
        big.color = cv2.GaussianBlur(big.color, (3, 3), 0)
        (obj,), _ = detect_hybrid(big.color, big.depth, K, big.d, [(grow(big.box_mask, 1), 0.9)],
                                  [(box_corners(big, grow=1.75), 0.9)], params)
        self.assertTrue(obj.ambiguous)  # judged on the box size
        self.assertAlmostEqual(obj.length_m, 0.0465)  # the reported size is still the known one

    def test_swollen_mask_with_a_normal_box_is_not_ambiguous(self):
        # a wall shadow next to the object swells the seg mask; the box shows one object
        s = self.scene
        shadow = grow(s.box_mask, 8) & ~s.box_mask
        s.color[shadow] = (40, 40, 40)
        s.depth[shadow] = s.hit(0.032)[shadow]  # about as high as the top face
        (obj,), _ = detect_hybrid(s.color, s.depth, K, s.d, [(grow(s.box_mask, 9), 0.9)],
                                  [(box_corners(s), 0.9)], PARAMS)
        self.assertGreater(obj.length_m, PARAMS.max_single_length)  # the seg size swelled
        self.assertFalse(obj.ambiguous)

    def test_far_off_seg_direction_replaced_near_the_box(self):
        (obj,) = self.run_hybrid([(box_corners(self.scene, 40), 0.9)])
        self.assertEqual(obj.angle_source, OBB_PEAK)
        # the new direction lies within the window of the box angle (image view of it, + margin)
        self.assertLessEqual(self.error(obj, 40), math.radians(PARAMS.hybrid_window_deg + 8))

    def test_box_without_a_mask_keeps_the_box_angle(self):
        (obj,) = self.run_hybrid([(box_corners(self.scene, 10), 0.9)], segments=[])
        self.assertEqual(obj.angle_source, OBB)
        self.assertIsNone(obj.mask_outline)
        self.assertAlmostEqual(obj.length_m, 0.046, delta=0.002)  # the box's own size
        self.assertFalse(obj.ambiguous)
        self.assertLess(self.error(obj, 10), math.radians(1))

    def test_masks_without_a_box_are_dropped(self):
        self.assertEqual(self.run_hybrid([]), [])


if __name__ == '__main__':
    unittest.main()
