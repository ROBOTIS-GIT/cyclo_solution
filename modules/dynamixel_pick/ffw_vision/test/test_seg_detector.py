from dataclasses import replace
import math
import unittest

import cv2
from ffw_vision.blob_detector import normalize_angle
from ffw_vision.seg_detector import detect_seg, fit_plane, project, resolve_model, SegParams
import numpy as np

WIDTH, HEIGHT = 424, 240
K = [218.0, 0.0, 211.0, 0.0, 218.0, 121.0, 0.0, 0.0, 1.0]
NO_DISTORTION = [0.0] * 5
D405 = [-0.0549, 0.0614, -0.0001, 0.0004, -0.0208]
TABLE = (230, 220, 210)  # BGR, low saturation
BLUE = (255, 140, 0)
DARK = (30, 30, 30)
BASKET = (90, 30, 340, 210)  # u0, v0, u1, v1
PARAMS = SegParams()


def rotation_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class Scene:
    """Camera above a table plane (tilted about x), a basket area, one box on the table."""

    def __init__(self, tilt_deg=0.0, cam_height=0.30, box=None, d=NO_DISTORTION):
        self.n = rotation_x(tilt_deg) @ np.array([0.0, 0.0, -1.0])  # toward the camera
        self.plane_d = cam_height
        self.d = d
        uu, vv = np.meshgrid(np.arange(WIDTH, dtype=np.float32),
                             np.arange(HEIGHT, dtype=np.float32))
        pixels = np.column_stack((uu.ravel(), vv.ravel())).reshape(-1, 1, 2)
        rays = cv2.undistortPoints(pixels, np.array(K).reshape(3, 3), np.array(d))
        rays = rays.reshape(HEIGHT, WIDTH, 2)
        self.rays = np.dstack((rays, np.ones((HEIGHT, WIDTH))))
        self.depth = self.hit(0.0)
        self.color = np.full((HEIGHT, WIDTH, 3), TABLE, np.uint8)
        u0, v0, u1, v1 = BASKET
        self.color[v0:v1, u0:u1] = BLUE
        self.box_mask = np.zeros((HEIGHT, WIDTH), bool)
        if box:
            self.add_box(*box)

    def hit(self, height):
        """Depth z of each pixel ray on the plane raised by height."""
        return ((height - self.plane_d) / (self.rays @ self.n)).astype(np.float32)

    def table_point(self, x, y):
        """3D point on the table; plane coordinates (m) start where the optical axis hits it."""
        a = np.cross(self.n, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(self.n, a)
        origin = np.array([0.0, 0.0, -self.plane_d / self.n[2]])
        return origin + x * a + y * b, a, b

    def add_box(self, x, y, length, width, deg, height):
        """Top face only: pixels whose ray meets the raised top inside the box."""
        _, a, b = self.table_point(0, 0)
        z_top = self.hit(height)
        p = self.rays * z_top[:, :, None]
        center = self.table_point(x, y)[0] + height * self.n
        axis = math.cos(math.radians(deg)) * a + math.sin(math.radians(deg)) * b
        rel = p - center
        across = np.cross(self.n, axis)
        inside = (np.abs(rel @ axis) <= length / 2) & (np.abs(rel @ across) <= width / 2)
        self.depth[inside] = z_top[inside]
        self.color[inside] = DARK
        self.box_mask |= inside
        self.center = center
        self.axis = axis


def grow(mask, px):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1,) * 2, np.uint8)).astype(bool)


class TestFitPlane(unittest.TestCase):

    def test_normal_points_to_camera(self):
        scene = Scene(tilt_deg=40)
        z = scene.depth
        points = (scene.rays * z[:, :, None]).reshape(-1, 3)
        n, d = fit_plane(points, PARAMS, np.random.default_rng(0))
        self.assertGreater(d, 0)
        self.assertAlmostEqual(d, 0.30, places=3)
        self.assertGreater(n @ scene.n, 0.999)


class TestDetectSeg(unittest.TestCase):

    def run_scene(self, scene, segments=None, params=PARAMS):
        if segments is None:
            segments = [(grow(scene.box_mask, 1), 0.9)]  # model masks bleed about 1 px
        return detect_seg(scene.color, scene.depth, K, scene.d, segments, params)

    def test_no_basket(self):
        scene = Scene()
        scene.color[:] = TABLE
        self.assertEqual(self.run_scene(scene, []), ([], None))

    def test_lying_box_straight_down(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 30, 0.034))
        scene.color = cv2.GaussianBlur(scene.color, (3, 3), 0)  # lens blur, no staircases
        (obj,), (n, d) = self.run_scene(scene)
        self.assertAlmostEqual(d, 0.30, places=3)
        self.assertAlmostEqual(obj.plane_height, 0.034, delta=0.002)
        self.assertEqual(obj.pose_label, 'lying')
        self.assertFalse(obj.ambiguous)
        # mask erosion shrinks the measured top face by about 1 px per side
        self.assertAlmostEqual(obj.length_m, 0.043, delta=0.003)
        self.assertAlmostEqual(obj.width_m, 0.031, delta=0.003)
        u, v = project([scene.center], K, scene.d)[0]
        self.assertAlmostEqual(obj.u, u, delta=1.0)
        self.assertAlmostEqual(obj.v, v, delta=1.0)
        (cu, cv), (tu, tv) = project([scene.center, scene.center + 0.02 * scene.axis], K, scene.d)
        expected = normalize_angle(math.atan2(tv - cv, tu - cu))
        self.assertAlmostEqual(normalize_angle(obj.angle - expected), 0.0, delta=math.radians(3))
        self.assertEqual(obj.angle_period, math.pi)
        self.assertAlmostEqual(obj.score, 0.9)

    def test_tilted_camera_and_distortion(self):
        for tilt, d in ((45, NO_DISTORTION), (45, D405)):
            with self.subTest(tilt=tilt, distorted=d is D405):
                box = (0.02, 0.01, 0.034, 0.028, 0, 0.047)
                scene = Scene(tilt_deg=tilt, cam_height=0.27, box=box, d=d)
                (obj,), _ = self.run_scene(scene)
                self.assertAlmostEqual(obj.plane_height, 0.047, delta=0.002)
                self.assertEqual(obj.pose_label, 'standing')
                u, v = project([scene.center], K, d)[0]
                self.assertAlmostEqual(obj.u, u, delta=1.0)
                self.assertAlmostEqual(obj.v, v, delta=1.0)
                self.assertEqual(obj.angle_period, math.pi / 2)  # 34 x 28 mm is near square

    def test_standing_top_is_square(self):
        scene = Scene(box=(0.0, 0.0, 0.040, 0.026, 0, 0.047))  # ratio above square_ratio
        (obj,), _ = self.run_scene(scene)
        self.assertEqual(obj.pose_label, 'standing')
        self.assertEqual(obj.angle_period, math.pi / 2)
        (obj,), _ = self.run_scene(scene, params=replace(PARAMS, standing_square=False))
        self.assertEqual(obj.angle_period, math.pi)

    def test_two_objects_as_one_mask_is_ambiguous(self):
        scene = Scene(box=(0.0, 0.0, 0.080, 0.034, 0, 0.034))
        (obj,), _ = self.run_scene(scene)
        self.assertTrue(obj.ambiguous)
        self.assertEqual(obj.pose_label, 'unknown')
        self.assertEqual(obj.angle_period, 0.0)

    def test_low_score_and_tiny_masks_dropped(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 0, 0.034))
        tiny = np.zeros_like(scene.box_mask)
        tiny[120:122, 200:202] = True
        objects, _ = self.run_scene(scene, [(grow(scene.box_mask, 1), 0.3), (tiny, 0.9)])
        self.assertEqual(objects, [])

    def test_blue_pixels_in_mask_are_ignored(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 0, 0.034))
        # mask bleeds 6 px onto the blue floor
        objects, _ = self.run_scene(scene, [(grow(scene.box_mask, 6), 0.9)])
        self.assertAlmostEqual(objects[0].length_m, 0.043, delta=0.003)

    def test_overlapping_masks_become_one_smaller_object(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 0, 0.034))
        wide = grow(scene.box_mask, 1)
        wide[:, :150] |= grow(scene.box_mask, 8)[:, :150]  # second mask also takes the wall side
        objects, _ = self.run_scene(scene, [(grow(scene.box_mask, 1), 0.65), (wide, 0.75)])
        (obj,) = objects
        self.assertAlmostEqual(obj.score, 0.75)
        self.assertAlmostEqual(obj.length_m, 0.043, delta=0.003)

    def test_edge_angle_in_tilted_distorted_view(self):
        for deg in (10, 37, 65, 120):
            with self.subTest(deg=deg):
                box = (0.02, 0.01, 0.046, 0.034, deg, 0.034)
                scene = Scene(tilt_deg=45, cam_height=0.27, box=box, d=D405)
                scene.color = cv2.GaussianBlur(scene.color, (3, 3), 0)  # lens blur, no staircases
                (obj,), _ = self.run_scene(scene)
                self.assertEqual(obj.angle_source, 'edges')
                (cu, cv), (tu, tv) = project([scene.center, scene.center + 0.02 * scene.axis],
                                             K, scene.d)
                expected = normalize_angle(math.atan2(tv - cv, tu - cu))
                self.assertAlmostEqual(normalize_angle(obj.angle - expected), 0.0,
                                       delta=math.radians(3))

    def test_blue_lines_under_a_bled_mask_do_not_vote(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 25, 0.034))
        bled = grow(scene.box_mask, 9)  # the model mask runs onto the basket wall
        ys, xs = np.nonzero(bled)
        shadow = (100, 50, 0)  # dark blue, still basket blue
        scene.color[ys.min():ys.max() + 1, xs.min() + 1:xs.min() + 5] = shadow  # wall line, 0 deg
        scene.color[ys.max() - 4:ys.max(), xs.min():xs.max() + 1] = shadow  # shadow line, 90 deg
        scene.color[scene.box_mask] = DARK
        scene.color = cv2.GaussianBlur(scene.color, (3, 3), 0)
        (cu, cv), (tu, tv) = project([scene.center, scene.center + 0.02 * scene.axis], K, scene.d)
        expected = normalize_angle(math.atan2(tv - cv, tu - cu))
        (obj,), _ = self.run_scene(scene, [(bled, 0.9)])
        self.assertAlmostEqual(normalize_angle(obj.angle - expected), 0.0, delta=math.radians(3))
        (old,), _ = self.run_scene(scene, [(bled, 0.9)], replace(PARAMS, edge_skip_blue=False))
        old_error = abs(normalize_angle(old.angle - expected, math.pi / 2))
        self.assertGreater(old_error, math.radians(5))

    def test_drawn_face_has_the_known_size(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 30, 0.030))  # lowest side down: top 30 mm
        params = replace(PARAMS, face_long=0.0465, face_mid=0.034, face_short=0.0285)
        (obj,), _ = self.run_scene(scene, params=params)
        corners = obj.face_corners.reshape(4, 2).astype(float)
        # sides of the drawn quad = the known 46.5 x 34 face seen at its height
        across = np.cross(scene.n, scene.axis)
        ref = project([scene.center + sa * 0.02325 * scene.axis + sb * 0.017 * across
                       for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))], K, scene.d)
        ref_sides = sorted(np.hypot(*(ref[i] - ref[(i + 1) % 4])) for i in range(4))
        sides = sorted(np.hypot(*(corners[i] - corners[(i + 1) % 4])) for i in range(4))
        np.testing.assert_allclose(sides, ref_sides, atol=2.0)
        (plain,), _ = self.run_scene(scene)  # no known size: the measured one is drawn
        self.assertIsNotNone(plain.face_corners)

    def test_debug_votes_leave_result_unchanged(self):
        scene = Scene(tilt_deg=30, box=(0.0, 0.0, 0.046, 0.034, 37, 0.034))
        scene.color = cv2.GaussianBlur(scene.color, (3, 3), 0)
        segments = [(grow(scene.box_mask, 1), 0.9)]
        plain, _ = detect_seg(scene.color, scene.depth, K, scene.d, segments, PARAMS)
        debug = []
        (obj,), _ = detect_seg(scene.color, scene.depth, K, scene.d, segments, PARAMS, None, debug)
        self.assertEqual((obj.u, obj.v, obj.angle, obj.length_m), (plain[0].u, plain[0].v,
                         plain[0].angle, plain[0].length_m))
        (info,) = debug
        self.assertEqual(len(info['edge_deg']), len(info['edge_u']))
        self.assertTrue(info['top'].any())
        self.assertIn('rect_deg', info)
        self.assertIn('peak_deg', info)

    def test_no_color_edges_falls_back_to_min_area_rect(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 30, 0.034))
        scene.color[grow(scene.box_mask, 6)] = DARK  # object and surroundings look the same
        (obj,), _ = self.run_scene(scene)
        self.assertEqual(obj.angle_source, 'min_area_rect')
        (obj,), _ = self.run_scene(Scene(box=(0.0, 0.0, 0.046, 0.034, 30, 0.034)),
                                   params=replace(PARAMS, angle_method='min_area_rect'))
        self.assertEqual(obj.angle_source, 'min_area_rect')

    def test_conf_parameter(self):
        scene = Scene(box=(0.0, 0.0, 0.046, 0.034, 0, 0.034))
        segments = [(grow(scene.box_mask, 1), 0.3)]
        objects, _ = self.run_scene(scene, segments, replace(PARAMS, conf=0.2))
        self.assertEqual(len(objects), 1)



class TestResolveModel(unittest.TestCase):

    def test_override_name_and_absolute_path(self):
        self.assertEqual(resolve_model('/tmp/best.pt', 'a.pt', '/share/models'), '/tmp/best.pt')
        self.assertEqual(resolve_model('', 'a.pt', '/share/models'), '/share/models/a.pt')
        self.assertEqual(resolve_model('', '/w/a.pt', '/share/models'), '/w/a.pt')
        with self.assertRaises(ValueError):
            resolve_model('', '', '/share/models')


if __name__ == '__main__':
    unittest.main()
