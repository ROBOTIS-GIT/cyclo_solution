from dataclasses import replace
import math
import unittest

import cv2
from ffw_vision.blob_detector import detect, DetectorParams, normalize_angle
import numpy as np

WIDTH, HEIGHT = 424, 240
BLUE = (255, 140, 0)  # BGR, saturated like the basket floor
DARK = (30, 30, 30)
FLOOR_M = 0.263
PARAMS = DetectorParams(roi=(120, 0, 340, 240))


def scene(boxes, tops=None, color=DARK, slope=0.0):
    """Blue floor with dark rotated boxes ((u, v), (length, width), deg); slope in m/px along u."""
    image = np.full((HEIGHT, WIDTH, 3), BLUE, np.uint8)
    floor = FLOOR_M + slope * (np.arange(WIDTH, dtype=np.float32) - WIDTH / 2)
    depth = np.tile(floor, (HEIGHT, 1)).astype(np.float32)
    for i, box in enumerate(boxes):
        region = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.fillPoly(region, [np.intp(cv2.boxPoints(box))], 1)
        image[region.astype(bool)] = color
        depth[region.astype(bool)] -= tops[i] if tops else 0.030
    return image, depth


def add_side_face(image, depth, top_box, side_px, height):
    """Dark strip left of an axis-aligned top face, depth ramping from top down to floor."""
    (u, v), (length, width), _ = top_box
    u_edge, v0, v1 = round(u - length / 2), round(v - width / 2), round(v + width / 2)
    for k in range(1, side_px + 1):
        col = u_edge - k
        image[v0:v1, col] = DARK
        depth[v0:v1, col] -= height * (1 - k / (side_px + 1))


class TestNormalizeAngle(unittest.TestCase):

    def test_range(self):
        for angle, expected in ((0.0, 0.0), (math.pi / 2, -math.pi / 2),
                                (-math.pi / 2, -math.pi / 2), (math.pi, 0.0),
                                (3 * math.pi / 4, -math.pi / 4)):
            with self.subTest(angle=angle):
                self.assertAlmostEqual(normalize_angle(angle), expected)


class TestDetect(unittest.TestCase):

    def test_empty_floor(self):
        image, depth = scene([])
        blobs, floor = detect(image, depth, PARAMS)
        self.assertEqual(blobs, [])
        self.assertAlmostEqual(floor, FLOOR_M, places=4)

    def test_rotated_lying_box(self):
        for deg in (0, 30, -60, 89):
            with self.subTest(deg=deg):
                image, depth = scene([((230, 120), (44, 32), deg)])
                blobs, _ = detect(image, depth, PARAMS)
                self.assertEqual(len(blobs), 1)
                blob = blobs[0]
                self.assertFalse(blob.ambiguous)
                self.assertEqual(blob.pose_label, 'lying')
                self.assertAlmostEqual(blob.u, 230, delta=1.0)
                self.assertAlmostEqual(blob.v, 120, delta=1.0)
                self.assertAlmostEqual(blob.length, 44, delta=2.0)
                self.assertAlmostEqual(blob.width, 32, delta=2.0)
                error = normalize_angle(blob.angle - math.radians(deg))
                self.assertLess(abs(error), math.radians(3))
                self.assertGreaterEqual(blob.angle, -math.pi / 2)
                self.assertLess(blob.angle, math.pi / 2)
                self.assertEqual(blob.angle_period, math.pi)
                self.assertAlmostEqual(blob.height, 0.030, places=3)
                self.assertAlmostEqual(blob.top_height, 0.030, places=3)

    def test_standing_is_near_square(self):
        image, depth = scene([((230, 120), (32, 29), 10)], tops=[0.047])
        blobs, _ = detect(image, depth, PARAMS)
        self.assertEqual(blobs[0].pose_label, 'standing')
        self.assertEqual(blobs[0].angle_period, math.pi / 2)

    def test_two_separate(self):
        image, depth = scene([((180, 80), (44, 32), 0), ((270, 170), (44, 32), 45)])
        blobs, _ = detect(image, depth, PARAMS)
        self.assertEqual(len(blobs), 2)
        self.assertFalse(any(b.ambiguous for b in blobs))

    def test_touching_is_ambiguous(self):
        image, depth = scene([((210, 120), (44, 32), 0), ((245, 120), (44, 32), 90)])
        blobs, _ = detect(image, depth, PARAMS)
        self.assertEqual(len(blobs), 1)
        self.assertTrue(blobs[0].ambiguous)
        self.assertEqual(blobs[0].pose_label, 'unknown')
        self.assertEqual(blobs[0].angle_period, 0.0)

    def test_cut_by_roi_edge_is_ambiguous(self):
        image, depth = scene([((230, 10), (44, 32), 0)])
        blobs, _ = detect(image, depth, PARAMS)
        self.assertTrue(blobs[0].ambiguous)

    def test_outside_roi_ignored(self):
        image, depth = scene([((60, 120), (44, 32), 0), ((380, 120), (44, 32), 0)])
        self.assertEqual(detect(image, depth, PARAMS)[0], [])

    def test_flat_dark_patch_dropped(self):
        image, depth = scene([((230, 120), (44, 32), 0)], tops=[0.0])
        self.assertEqual(detect(image, depth, PARAMS)[0], [])

    def test_white_rim_ignored(self):
        image, depth = scene([((230, 120), (44, 32), 0)], color=(240, 240, 240))
        self.assertEqual(detect(image, depth, PARAMS)[0], [])

    def test_small_noise_ignored(self):
        image, depth = scene([((230, 120), (10, 10), 0)])
        self.assertEqual(detect(image, depth, PARAMS)[0], [])

    def test_tilted_floor_uses_local_floor(self):
        slope = -0.0001  # left is farther, ~4.7 deg at 0.27 m as measured on the robot
        image, depth = scene([((150, 120), (32, 28), 0), ((300, 120), (44, 32), 0)],
                             tops=[0.047, 0.030], slope=slope)
        blobs = sorted(detect(image, depth, PARAMS)[0], key=lambda b: b.u)
        self.assertEqual([b.pose_label for b in blobs], ['standing', 'lying'])
        self.assertAlmostEqual(blobs[0].height, 0.047, delta=0.002)
        self.assertAlmostEqual(blobs[1].height, 0.030, delta=0.002)

    def test_side_face_is_left_out_of_the_box(self):
        top_box = ((230, 120), (34, 28), 0)
        image, depth = scene([top_box], tops=[0.047])
        add_side_face(image, depth, top_box, 12, 0.047)
        blob = detect(image, depth, PARAMS)[0][0]
        self.assertEqual(blob.pose_label, 'standing')
        self.assertAlmostEqual(blob.u, 230, delta=1.5)
        self.assertAlmostEqual(blob.length, 34, delta=3.0)
        self.assertLess(blob.color_box[0], 226)  # whole blob is pulled toward the side face

    def test_ambiguous_keeps_color_box(self):
        image, depth = scene([((210, 120), (44, 32), 0), ((245, 120), (44, 32), 90)],
                             tops=[0.030, 0.047])
        blob = detect(image, depth, PARAMS)[0][0]
        self.assertEqual((blob.u, blob.v, blob.length), blob.color_box[:3])

    def test_without_depth(self):
        image, _ = scene([((230, 120), (44, 32), 0)])
        blobs, floor = detect(image, None, PARAMS)
        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0].pose_label, 'unknown')
        self.assertTrue(math.isnan(blobs[0].height) and math.isnan(floor))

    def test_invalid_depth_gives_unknown(self):
        image, depth = scene([((230, 120), (44, 32), 0)])
        depth[90:150, 200:260] = np.nan
        blobs, _ = detect(image, depth, PARAMS)
        self.assertEqual(blobs[0].pose_label, 'unknown')

    def test_full_image_roi(self):
        image, depth = scene([((60, 120), (44, 32), 0)])
        blobs, _ = detect(image, depth, replace(PARAMS, roi=(0, 0, 0, 0)))
        self.assertEqual(len(blobs), 1)


if __name__ == '__main__':
    unittest.main()
