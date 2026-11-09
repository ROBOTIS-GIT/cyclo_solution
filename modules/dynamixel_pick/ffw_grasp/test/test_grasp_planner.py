from dataclasses import replace
import math
import unittest

import cv2
from ffw_grasp.grasp_planner import (
    away_roll_sign, backup_rolls, Basket, fit_basket, GraspParams, matrix_to_quat, merge_planes,
    merge_targets, near_any, pick_object, pick_target, plan_grasp, quat_to_matrix, rank_objects,
    Target, top_face, UP, veil_pixels)
from ffw_vision.blob_detector import normalize_angle
from ffw_vision.seg_detector import pixel_rays, project
import numpy as np

WIDTH, HEIGHT = 424, 240
K = [218.0, 0.0, 211.0, 0.0, 218.0, 121.0, 0.0, 0.0, 1.0]
D405 = [-0.0549, 0.0614, -0.0001, 0.0004, -0.0208]
TABLE_Z = 0.72
PARAMS = GraspParams()


def rotation_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class Scene:
    """Camera above a table tilted about x; plane coordinates start at the optical axis hit."""

    def __init__(self, tilt_deg=46.0, cam_height=0.27, d=D405):
        self.n = rotation_x(tilt_deg) @ np.array([0.0, 0.0, -1.0])  # up, camera frame
        self.plane_d = cam_height
        self.d = d
        rays = pixel_rays(WIDTH, HEIGHT, K, d)
        self.rays = rays
        self.rays3 = np.dstack((rays, np.ones((HEIGHT, WIDTH))))
        self.a = np.cross(self.n, [1.0, 0.0, 0.0])
        self.a /= np.linalg.norm(self.a)
        self.b = np.cross(self.n, self.a)
        self.origin = np.array([0.0, 0.0, -cam_height / self.n[2]])
        self.depth = self.hit(0.0)
        # base frame: x = a, y = b, z = up; table at TABLE_Z
        self.rotation = np.vstack((self.a, self.b, self.n))
        self.translation = np.array([0.3, -0.2, TABLE_Z]) - self.rotation @ self.origin

    def hit(self, height):
        return ((height - self.plane_d) / (self.rays3 @ self.n)).astype(np.float32)

    def point(self, x, y, height=0.0):
        return self.origin + x * self.a + y * self.b + height * self.n

    def direction(self, deg):
        return math.cos(math.radians(deg)) * self.a + math.sin(math.radians(deg)) * self.b

    def raise_box(self, x, y, length, width, deg, height):
        """Depth of the top face of a box (sides are not drawn)."""
        z = self.hit(height)
        rel = self.rays3 * z[:, :, None] - self.point(x, y, height)
        axis = self.direction(deg)
        inside = ((np.abs(rel @ axis) <= length / 2)
                  & (np.abs(rel @ np.cross(self.n, axis)) <= width / 2))
        self.depth[inside] = z[inside]

    def target(self, x, y, length, width, deg, height, period=math.pi, pose='lying'):
        """Observation as Vision makes it: projected center and spans between side ends."""
        center = self.point(x, y, height)
        half_long = self.direction(deg) * length / 2
        half_short = np.cross(self.n, self.direction(deg)) * width / 2
        (u, v), tip, tail, side, other = project(
            [center, center + half_long, center - half_long, center + half_short,
             center - half_short], K, self.d)
        angle = normalize_angle(math.atan2(tip[1] - v, tip[0] - u))
        return Target(u, v, np.hypot(*(tip - tail)), np.hypot(*(side - other)),
                      angle, period, pose, height, 0.8)

    def plan(self, target, ee_y=(0.0, 1.0, 0.0), params=PARAMS, samples=None):
        return plan_grasp(target, (self.n, self.plane_d), self.depth, K, self.d, self.rays,
                          self.rotation, self.translation, np.array(ee_y), params, samples)

    def to_base(self, p):
        return self.rotation @ p + self.translation


LYING = (0.02, 0.01, 0.046, 0.034, 30.0, 0.034)


class TestTopFace(unittest.TestCase):

    def test_round_trip_through_pixels(self):
        for d in ([0.0] * 5, D405):
            with self.subTest(distorted=d is D405):
                scene = Scene(d=d)
                x, y, length, width, deg, height = LYING
                center, axis, l_m, w_m = top_face(
                    scene.target(*LYING), K, d, scene.n, scene.plane_d)
                self.assertLess(np.linalg.norm(center - scene.point(x, y, height)), 0.0003)
                self.assertGreater(abs(axis @ scene.direction(deg)), math.cos(math.radians(0.5)))
                self.assertAlmostEqual(l_m, length, delta=0.0005)
                self.assertAlmostEqual(w_m, width, delta=0.0005)


class TestPlanGrasp(unittest.TestCase):

    def test_free_box_closes_across_the_short_side(self):
        scene = Scene()
        scene.raise_box(*LYING)
        g = scene.plan(scene.target(*LYING))
        self.assertEqual(g.chosen.name, 'A')
        self.assertAlmostEqual(g.chosen.opening, 0.034 + 2 * 0.012, delta=0.001)
        x, y, _, _, deg, height = LYING
        expected = scene.to_base(scene.point(x, y, height - 0.015))
        self.assertLess(np.linalg.norm(g.position - expected), 0.001)
        self.assertAlmostEqual(g.position[2], TABLE_Z + 0.034 - 0.015, delta=0.001)
        self.assertAlmostEqual(g.table_z, TABLE_Z, delta=0.0005)
        np.testing.assert_allclose(g.rotation[:, 2], UP)
        short = scene.rotation @ np.cross(scene.n, scene.direction(deg))
        self.assertGreater(abs(g.rotation[:, 1] @ short), 0.999)
        self.assertGreaterEqual(g.rotation[:, 1] @ [0.0, 1.0, 0.0], 0.0)
        np.testing.assert_allclose(g.rotation.T @ g.rotation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(g.rotation), 1.0)
        np.testing.assert_allclose(g.hover, g.position + [0.0, 0.0, 0.10])

    def test_yaw_closer_to_the_current_end_effector(self):
        scene = Scene()
        scene.raise_box(*LYING)
        target = scene.target(*LYING)
        y1 = scene.plan(target, ee_y=(0.0, 1.0, 0.0)).rotation[:, 1]
        g = scene.plan(target, ee_y=(0.0, -1.0, 0.0))
        np.testing.assert_allclose(g.rotation[:, 1], -y1, atol=1e-9)
        self.assertLessEqual(abs(g.wrist_turn), math.pi / 2)

    def test_wall_beside_the_long_side_switches_to_b(self):
        scene = Scene()
        scene.raise_box(*LYING)
        x, y, _, width, deg, _ = LYING
        # wall where the A fingers go: 10 mm beyond the object's long side
        side = np.array([math.cos(math.radians(deg + 90)), math.sin(math.radians(deg + 90))])
        wx, wy = np.array([x, y]) + side * (width / 2 + 0.010 + 0.010)
        scene.raise_box(wx, wy, 0.120, 0.020, deg, 0.075)
        g = scene.plan(scene.target(*LYING))
        a, b = g.candidates
        self.assertTrue(a.reason.startswith('blocked'))
        self.assertEqual(g.chosen.name, 'B')
        self.assertAlmostEqual(g.chosen.opening, 0.046 + 2 * 0.012, delta=0.001)

    def test_boxed_in_gives_no_grasp(self):
        scene = Scene()
        x, y, length, width, deg, height = LYING
        for wall_deg, offset, wall_length in ((deg + 90, width / 2 + 0.02, 0.12),
                                              (deg, length / 2 + 0.02, 0.12)):
            side = np.array([math.cos(math.radians(wall_deg)), math.sin(math.radians(wall_deg))])
            for sign in (1, -1):
                wx, wy = np.array([x, y]) + sign * side * (offset + 0.01)
                scene.raise_box(wx, wy, wall_length, 0.02, wall_deg - 90, 0.075)
        scene.raise_box(*LYING)
        g = scene.plan(scene.target(*LYING))
        self.assertIsNone(g.chosen)
        self.assertIsNone(g.position)
        self.assertIn('A blocked', g.reason)
        self.assertIn('B blocked', g.reason)

    def test_near_square_opens_for_the_long_side(self):
        scene = Scene()
        box = (0.0, 0.0, 0.034, 0.0285, 10.0, 0.050)
        scene.raise_box(*box)
        g = scene.plan(scene.target(*box, period=math.pi / 2, pose='standing'))
        self.assertEqual(g.chosen.name, 'A')
        self.assertAlmostEqual(g.chosen.grip, 0.034, delta=0.0005)
        self.assertAlmostEqual(g.position[2], TABLE_Z + 0.050 - 0.015, delta=0.001)

    def test_near_square_frames_swapping_sides(self):
        scene = Scene()
        box = (0.0, 0.0, 0.034, 0.0285, 10.0, 0.050)
        scene.raise_box(*box)
        # some frames measure the other side as the long one
        frames = [scene.target(*box, period=math.pi / 2, pose='standing')] * 3 + [
            scene.target(0.0, 0.0, 0.031, 0.030, 100.0, 0.050, math.pi / 2, 'standing')] * 2
        target, kept = merge_targets(frames, 3.0, 5)
        g = scene.plan(target, samples=[frames[i] for i in kept])
        self.assertAlmostEqual(g.chosen.grip, 0.034, delta=0.0005)
        self.assertGreater(abs(g.axis @ scene.direction(10.0)), math.cos(math.radians(1)))

    def test_known_size_widens_an_eroded_measurement(self):
        scene = Scene()
        real = (0.0, 0.0, 0.0465, 0.034, 0.0, 0.034)
        scene.raise_box(*real)
        eroded = scene.target(0.0, 0.0, 0.038, 0.024, 0.0, 0.034)
        known = replace(PARAMS, known_short=0.034, known_long=0.0465)
        a, b = scene.plan(eroded, params=known).candidates
        self.assertAlmostEqual(a.opening, 0.034 + 2 * 0.012, delta=1e-6)
        self.assertAlmostEqual(b.opening, 0.0465 + 2 * 0.012, delta=1e-6)
        a, b = scene.plan(eroded).candidates  # off: measured size
        self.assertAlmostEqual(a.grip, 0.024, delta=0.0005)
        big = scene.target(0.0, 0.0, 0.050, 0.038, 0.0, 0.034)  # larger measurement wins
        a, b = scene.plan(big, params=known).candidates
        self.assertAlmostEqual(a.grip, 0.038, delta=0.0005)
        self.assertAlmostEqual(b.grip, 0.050, delta=0.0005)
        square = scene.target(0.0, 0.0, 0.030, 0.026, 0.0, 0.050, math.pi / 2, 'standing')
        a, b = scene.plan(square, params=known).candidates
        self.assertEqual((a.grip, b.grip), (0.034, 0.034))

    def test_near_square_takes_the_least_wrist_turn(self):
        scene = Scene()
        box = (0.0, 0.0, 0.034, 0.0285, 10.0, 0.050)
        scene.raise_box(*box)
        target = scene.target(*box, period=math.pi / 2, pose='standing')
        self.assertEqual(scene.plan(target, ee_y=(0.0, 1.0, 0.0)).chosen.name, 'A')
        g = scene.plan(target, ee_y=(1.0, 0.0, 0.0))
        self.assertEqual(g.chosen.name, 'B')
        self.assertLess(abs(g.wrist_turn), math.radians(45))

    def test_least_wrist_turn_wins(self):
        scene = Scene()
        scene.raise_box(*LYING)  # long side at 30 deg
        g = scene.plan(scene.target(*LYING), ee_y=(1.0, 0.0, 0.0))
        self.assertEqual(g.chosen.name, 'B')
        self.assertLess(abs(g.wrist_turn), math.radians(45))
        self.assertEqual(scene.plan(scene.target(*LYING)).chosen.name, 'A')

    def test_end_effector_above_the_fingertip(self):
        scene = Scene()
        scene.raise_box(*LYING)
        g = scene.plan(scene.target(*LYING), params=replace(PARAMS, tip_below_end_effector=0.05))
        np.testing.assert_allclose(g.position - g.tip, [0.0, 0.0, 0.05], atol=1e-9)
        np.testing.assert_allclose(g.hover - g.tip, [0.0, 0.0, 0.15], atol=1e-9)
        self.assertAlmostEqual(g.tip[2], TABLE_Z + 0.034 - 0.015, delta=0.001)

    def test_tilt_leans_the_top_toward_the_robot(self):
        scene = Scene()
        box = (0.02, 0.01, 0.0465, 0.034, 0.0, 0.034)  # long side along base x: A closes along y
        scene.raise_box(*box)
        tilted = replace(PARAMS, max_tilt_deg=50.0, tip_below_end_effector=0.05)
        g = scene.plan(scene.target(*box), params=tilted)
        self.assertEqual(g.chosen.name, 'A')
        self.assertAlmostEqual(math.degrees(g.chosen.tilt), 50.0, delta=1e-6)
        s50, c50 = math.sin(math.radians(50)), math.cos(math.radians(50))
        np.testing.assert_allclose(g.rotation[:, 2], [-s50, 0.0, c50], atol=1e-3)
        self.assertAlmostEqual(abs(g.rotation[1, 1]), 1.0, delta=1e-3)  # closes along base y
        np.testing.assert_allclose(g.rotation.T @ g.rotation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(g.rotation), 1.0)
        np.testing.assert_allclose(g.position - g.tip, 0.05 * g.rotation[:, 2], atol=1e-9)
        np.testing.assert_allclose(g.hover - g.position, 0.10 * g.rotation[:, 2], atol=1e-9)
        x, y, _, _, _, height = box
        # the tilted end face dips half the finger width: the tip rises to keep the corner up
        tip_height = max(height - 0.015, 0.012 + 0.015 * s50)
        expected = scene.to_base(scene.point(x, y, tip_height))
        self.assertLess(np.linalg.norm(g.tip - expected), 0.001)

    def test_no_tilt_when_the_closing_axis_points_at_the_robot(self):
        scene = Scene()
        box = (0.02, 0.01, 0.0465, 0.034, 90.0, 0.034)  # A closes along base x
        scene.raise_box(*box)
        g = scene.plan(scene.target(*box), params=replace(PARAMS, max_tilt_deg=50.0))
        a, b = g.candidates
        self.assertAlmostEqual(a.tilt, 0.0, delta=math.radians(0.1))
        self.assertIs(g.chosen, b)  # closes along y: no wrist turn, full tilt
        self.assertAlmostEqual(math.degrees(b.tilt), 50.0, delta=0.1)

    def test_wall_on_the_lean_side_reduces_the_tilt(self):
        scene = Scene()
        box = (0.02, 0.01, 0.0465, 0.034, 0.0, 0.034)
        scene.raise_box(*box)
        scene.raise_box(0.02 - 0.060, 0.01, 0.12, 0.02, 90.0, 0.075)  # 50-70 mm toward the robot
        g = scene.plan(scene.target(*box), params=replace(PARAMS, max_tilt_deg=50.0))
        self.assertEqual(g.chosen.name, 'A')
        self.assertAlmostEqual(math.degrees(g.chosen.tilt), 30.0, delta=1e-6)
        g = scene.plan(scene.target(*box))  # vertical fingers do not reach the wall
        self.assertEqual(g.chosen.blocked, 0)

    def test_options_list_every_free_tilt_chosen_first(self):
        scene = Scene()
        box = (0.02, 0.01, 0.0465, 0.034, 0.0, 0.034)
        scene.raise_box(*box)
        g = scene.plan(scene.target(*box), params=replace(PARAMS, max_tilt_deg=40.0))
        first = g.options[0]
        self.assertEqual((first.name, first.tilt), (g.chosen.name, g.chosen.tilt))
        np.testing.assert_allclose(first.position, g.position)
        np.testing.assert_allclose(first.rotation, g.rotation)
        tilts = [round(math.degrees(o.tilt)) for o in g.options if o.name == 'A']
        self.assertEqual(tilts, [40, 30, 20, 10, 0])
        names = [o.name for o in g.options]
        self.assertEqual(names, sorted(names, key=lambda n: n != 'A'))  # A (no turn) before B
        for o in g.options:
            np.testing.assert_allclose(o.rotation.T @ o.rotation, np.eye(3), atol=1e-9)
            np.testing.assert_allclose(o.hover - o.position, 0.10 * o.rotation[:, 2], atol=1e-9)

    def test_options_skip_blocked_tilts(self):
        scene = Scene()
        box = (0.02, 0.01, 0.0465, 0.034, 0.0, 0.034)
        scene.raise_box(*box)
        scene.raise_box(0.02 - 0.060, 0.01, 0.12, 0.02, 90.0, 0.075)  # wall on the lean side
        g = scene.plan(scene.target(*box), params=replace(PARAMS, max_tilt_deg=50.0))
        tilts = [round(math.degrees(o.tilt)) for o in g.options if o.name == 'A']
        self.assertEqual(tilts[0], 30)
        self.assertNotIn(50, tilts)
        self.assertNotIn(40, tilts)

    def test_lowest_finger_corner_stays_above_the_floor(self):
        scene = Scene()
        box = (0.02, 0.01, 0.0465, 0.034, 0.0, 0.0335)  # 28.5 mm on a 5 mm floor
        scene.raise_box(*box)
        params = replace(PARAMS, max_tilt_deg=50.0, finger_width=0.015, floor_height=0.005,
                         floor_clearance=0.005, close_drop=0.006)
        g = scene.plan(scene.target(*box), params=params)
        t = g.chosen.tilt
        lowest = g.chosen.tip_height - 0.0075 * math.sin(t) - 0.006 * math.cos(t)
        self.assertAlmostEqual(lowest, 0.010, delta=1e-6)
        self.assertAlmostEqual(g.tip[2], TABLE_Z + g.chosen.tip_height, delta=0.001)

    def test_unsure_long_side_opens_for_the_long_one(self):
        scene = Scene()
        box = (0.02, 0.01, 0.0465, 0.034, 0.0, 0.034)
        scene.raise_box(*box)
        eroded = scene.target(0.02, 0.01, 0.037, 0.032, 0.0, 0.034)  # looks almost square
        params = replace(PARAMS, known_short=0.034, known_long=0.0465, long_side_poses='lying')
        a, b = scene.plan(eroded, params=params).candidates
        self.assertEqual((a.grip, b.grip), (0.0465, 0.0465))
        a, b = scene.plan(eroded, params=replace(params, long_side_poses='')).candidates
        self.assertEqual(a.grip, 0.034)

    def test_wrist_link_on_the_lean_side_reduces_the_tilt(self):
        scene = Scene()
        box = (0.02, 0.01, 0.0465, 0.034, 0.0, 0.034)
        scene.raise_box(*box)
        scene.raise_box(0.02 - 0.15, 0.01, 0.20, 0.02, 90.0, 0.075)  # 14-16 cm toward the robot
        tilted = replace(PARAMS, max_tilt_deg=50.0, finger_width=0.015)
        self.assertAlmostEqual(math.degrees(scene.plan(scene.target(*box), params=tilted)
                                            .chosen.tilt), 50.0, delta=1e-6)
        linked = replace(tilted, finger_length=0.10, link_start=0.155, link_half_y=0.024,
                         link_x_min=-0.058, link_x_max=0.133)
        g = scene.plan(scene.target(*box), params=linked)
        self.assertLess(math.degrees(g.chosen.tilt), 50.0 - 1e-6)
        self.assertLess(g.chosen.blocked, PARAMS.obstacle_points)

    def test_low_object_keeps_the_tip_above_the_table(self):
        scene = Scene()
        box = (0.0, 0.0, 0.046, 0.034, 0.0, 0.020)
        scene.raise_box(*box)
        g = scene.plan(scene.target(*box))
        self.assertAlmostEqual(g.position[2], TABLE_Z + 0.012, delta=0.001)

    def test_too_wide_and_z_biases(self):
        scene = Scene()
        scene.raise_box(*LYING)
        g = scene.plan(scene.target(*LYING), params=replace(PARAMS, max_open=0.050))
        self.assertEqual(g.candidates[1].reason, 'too wide')
        self.assertEqual(g.chosen.name, 'A')
        g1 = scene.plan(scene.target(*LYING))
        g2 = scene.plan(scene.target(*LYING), params=replace(PARAMS, camera_z_bias=0.016))
        np.testing.assert_allclose(g2.tip - g1.tip, [0.0, 0.0, -0.016], atol=1e-9)
        np.testing.assert_allclose(g2.hover - g1.hover, [0.0, 0.0, -0.016], atol=1e-9)
        g4 = scene.plan(scene.target(*LYING),
                        params=replace(PARAMS, camera_x_bias=-0.008, camera_y_bias=0.003))
        np.testing.assert_allclose(g4.tip - g1.tip, [0.008, -0.003, 0.0], atol=1e-9)
        np.testing.assert_allclose(g4.hover - g1.hover, [0.008, -0.003, 0.0], atol=1e-9)
        g3 = scene.plan(scene.target(*LYING), params=replace(PARAMS, arm_z_bias=0.005))
        np.testing.assert_allclose(g3.tip, g1.tip, atol=1e-9)  # the target itself stays
        np.testing.assert_allclose(g3.position - g1.position, [0.0, 0.0, -0.005], atol=1e-9)


def target(u=100.0, v=50.0, angle=0.3, period=math.pi, pose='lying', height=0.034,
           score=0.8):
    return Target(u, v, 30.0, 20.0, angle, period, pose, height, score)


class TestVeil(unittest.TestCase):

    def mixed(self, scene):
        """Mixed depth 2 px past the box edge: rays land 21 mm high, beyond the far side."""
        raised = scene.depth < scene.hit(0.0) - 0.005
        grown = cv2.dilate(raised.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ring = grown & ~raised
        scene.depth[ring] = scene.hit(0.021)[ring]
        return raised, ring

    def test_only_the_jump_is_flagged(self):
        scene = Scene()
        working = scene.depth < 0.5  # the far table grazes away: depth changes fast per px
        self.assertFalse((veil_pixels(scene.depth, 2, 0.008) & working).any())
        scene.raise_box(*LYING)
        raised, ring = self.mixed(scene)
        veil = veil_pixels(scene.depth, 2, 0.008)
        self.assertGreater((veil & ring).sum(), 0.5 * ring.sum())
        inner = cv2.erode(raised.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        self.assertFalse((veil & inner).any())  # the top face stays

    def test_mixed_depth_behind_an_edge_does_not_block(self):
        scene = Scene()
        scene.raise_box(*LYING)
        target = scene.target(*LYING)
        self.mixed(scene)
        g = scene.plan(target)
        self.assertLess(max(c.blocked for c in g.candidates), PARAMS.obstacle_points)
        self.assertEqual(g.chosen.name, 'A')
        g = scene.plan(target, params=replace(PARAMS, veil_jump=1.0))  # filter off
        self.assertGreaterEqual(max(c.blocked for c in g.candidates), PARAMS.obstacle_points)


class TestMerge(unittest.TestCase):

    def test_pick_highest_score_with_height(self):
        self.assertIsNone(pick_target([]))
        best = pick_target([target(score=0.9, height=float('nan')), target(score=0.7),
                            target(u=10.0, score=0.8)])
        self.assertEqual(best.u, 10.0)

    def test_pick_object_keeps_one_object_when_the_best_flips(self):
        # two objects with close scores: the single-frame best changes between them
        frames = [[target(u=100.0, score=0.80 + 0.02 * (i % 2)),
                   target(u=200.0, score=0.81 - 0.02 * (i % 2))] for i in range(10)]
        self.assertEqual({pick_target(f).u for f in frames}, {100.0, 200.0})
        targets, which, seen = pick_object(frames, 10.0, 7)
        self.assertEqual({t.u for t in targets}, {100.0})  # median 0.81 vs 0.80
        self.assertEqual(which, list(range(10)))
        self.assertEqual(seen, 2)
        merged, kept = merge_targets(targets, 3.0, 7)
        self.assertEqual(len(kept), 10)

    def test_pick_object_follows_jitter_and_needs_min_frames(self):
        frames = [[target(u=100.0 + 0.5 * i, score=0.7)] for i in range(10)]
        for i in range(3):  # a higher score seen in 3 frames only
            frames[i].append(target(u=300.0, score=0.95))
        frames[4].append(target(u=150.0, height=float('nan'), score=0.99))  # unmeasured
        targets, which, seen = pick_object(frames, 10.0, 7)
        self.assertEqual(len(targets), 10)
        self.assertTrue(all(t.score == 0.7 for t in targets))
        self.assertEqual(seen, 1)
        with self.assertRaises(ValueError):
            pick_object([[], []], 10.0, 1)

    def test_pick_object_two_close_targets_stay_apart(self):
        # touching objects 8 px apart: one target per object and frame
        frames = [[target(u=100.0, score=0.6), target(u=108.0, score=0.9)] for _ in range(8)]
        targets, which, seen = pick_object(frames, 10.0, 7)
        self.assertEqual({t.u for t in targets}, {108.0})
        self.assertEqual(len(targets), 8)
        self.assertEqual(seen, 2)

    def test_rank_objects_best_first(self):
        # 0.9 seen in 3 frames only goes after the ones seen in 7 or more
        frames = [[target(u=100.0, score=0.6), target(u=200.0, score=0.8)] for _ in range(8)]
        for i in range(3):
            frames[i].append(target(u=300.0, score=0.9))
        ranked = rank_objects(frames, 10.0, 7)
        self.assertEqual([targets[0].u for targets, _ in ranked], [200.0, 100.0, 300.0])
        self.assertEqual(ranked[2][1], [0, 1, 2])
        self.assertEqual(pick_object(frames, 10.0, 7)[0], ranked[0][0])

    def test_near_any(self):
        self.assertFalse(near_any((0.5, -0.2), [], 0.025))
        self.assertTrue(near_any((0.5, -0.2), [(0.6, 0.0), (0.52, -0.21)], 0.025))
        self.assertFalse(near_any((0.5, -0.2), [(0.53, -0.21)], 0.025))

    def test_outlier_frame_dropped(self):
        frames = [target(u=100.0 + 0.1 * i) for i in range(9)] + [target(u=130.0)]
        merged, kept = merge_targets(frames, 3.0, 7)
        self.assertEqual(kept, list(range(9)))
        self.assertAlmostEqual(merged.u, 100.4)

    def test_minority_pose_dropped(self):
        frames = [target() for _ in range(8)] + [target(pose='standing', period=math.pi / 2)] * 2
        merged, kept = merge_targets(frames, 3.0, 7)
        self.assertEqual(len(kept), 8)
        self.assertEqual(merged.pose_label, 'lying')
        with self.assertRaises(ValueError):
            merge_targets(frames, 3.0, 9)

    def test_pose_from_median_height(self):
        # tilted object near the threshold: labels flip, the median height decides once
        lying = [target(height=0.040) for _ in range(5)]
        standing = [target(pose='standing', period=math.pi / 2, height=0.043)] * 5
        merged, kept = merge_targets(lying + standing, 3.0, 7, 0.042)
        self.assertEqual(len(kept), 10)
        self.assertEqual(merged.pose_label, 'lying')
        self.assertEqual(merged.angle_period, math.pi / 2)  # long side unsure: open wide
        merged, kept = merge_targets(lying[:3] + standing, 3.0, 7, 0.042)
        self.assertEqual(len(kept), 8)
        self.assertEqual(merged.pose_label, 'standing')

    def test_square_flicker_is_kept_as_unsure(self):
        frames = [target() for _ in range(6)] + [target(period=math.pi / 2)] * 4
        merged, kept = merge_targets(frames, 3.0, 7)
        self.assertEqual(len(kept), 10)
        self.assertEqual(merged.angle_period, math.pi / 2)
        with self.assertRaisesRegex(ValueError, '3 moved over 3 px, 0 with another pose'):
            merge_targets([target()] * 5 + [target(u=150.0)] * 3, 3.0, 7)

    def test_angle_mean_across_the_wrap(self):
        half = math.pi / 2
        frames = [target(angle=a) for a in (half - 0.02, -half + 0.02, half - 0.01, -half)]
        merged, _ = merge_targets(frames, 3.0, 4)
        self.assertAlmostEqual(normalize_angle(merged.angle - half), 0.0, delta=0.01)
        frames = [target(angle=a, period=math.pi / 2) for a in (0.77, -0.77, 0.78)]
        merged, _ = merge_targets(frames, 3.0, 3)
        self.assertAlmostEqual(abs(math.cos(4 * merged.angle) + 1), 0.0, delta=0.01)

    def test_no_orientation_or_frames(self):
        with self.assertRaises(ValueError):
            merge_targets([], 3.0, 1)
        with self.assertRaises(ValueError):
            merge_targets([target(period=0.0)] * 3, 3.0, 3)

    def test_merge_planes(self):
        n, d = merge_planes([[0.0, 0.0, -2.0, 0.3], [0.0, 0.1, -2.0, 0.31],
                             [0.0, -0.1, -2.0, 0.29]])
        np.testing.assert_allclose(n, [0.0, 0.0, -1.0])
        self.assertAlmostEqual(d, 0.3)


BLUE = (255, 128, 0)  # BGR, HSV hue 105
BASKET = replace(PARAMS, basket_length=0.32, basket_width=0.24, basket_height=0.075)


def basket_scene(x, y, deg, cam_height=0.55):
    """Blue basket rim (15 mm wide) and floor on the table; walls are not drawn."""
    scene = Scene(cam_height=cam_height)
    color = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    axis = scene.direction(deg)
    for height, inset_out, inset_in in ((0.005, 0.015, 1.0), (0.075, 0.0, 0.015)):
        z = scene.hit(height)
        rel = scene.rays3 * z[:, :, None] - scene.point(x, y, height)
        along, across = np.abs(rel @ axis), np.abs(rel @ np.cross(scene.n, axis))
        inside = (along <= 0.16 - inset_out) & (across <= 0.12 - inset_out)
        if inset_in < 1.0:
            inside &= (along > 0.16 - inset_in) | (across > 0.12 - inset_in)
        scene.depth[inside] = z[inside]
        color[inside] = BLUE
    return scene, color


class TestBasket(unittest.TestCase):

    def fit(self, scene, color):
        return fit_basket(color, scene.depth, K, scene.d, scene.rays, (scene.n, scene.plane_d),
                          BASKET)

    def test_whole_basket(self):
        scene, color = basket_scene(0.0, 0.02, 20.0)
        basket, reason = self.fit(scene, color)
        self.assertEqual(reason, '')
        self.assertEqual(basket.cut, '')
        self.assertLess(np.linalg.norm(basket.center - scene.point(0.0, 0.02)), 0.004)
        self.assertGreater(abs(basket.axis @ scene.direction(20.0)), math.cos(math.radians(1)))
        self.assertAlmostEqual(basket.length_m, 0.32, delta=0.008)
        self.assertAlmostEqual(basket.width_m, 0.24, delta=0.008)

    def test_cut_side_placed_from_the_visible_side(self):
        # long side runs off the image: the far end comes from the known length
        scene, color = basket_scene(0.0, 0.33, 90.0, cam_height=0.30)
        basket, reason = self.fit(scene, color)
        self.assertEqual(reason, '')
        self.assertEqual(basket.cut, 'long')
        self.assertLess(basket.length_m, 0.30)
        self.assertLess(np.linalg.norm(basket.center - scene.point(0.0, 0.33)), 0.004)
        self.assertGreater(abs(basket.axis @ scene.direction(90.0)), math.cos(math.radians(1)))

    def test_no_basket(self):
        scene = Scene()
        basket, reason = self.fit(scene, np.zeros((HEIGHT, WIDTH, 3), np.uint8))
        self.assertIsNone(basket)
        self.assertEqual(reason, 'basket not seen')
        self.assertEqual(fit_basket(None, None, K, D405, None, None, PARAMS), (None, 'basket off'))


class TestQuaternion(unittest.TestCase):

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        quats = [q / np.linalg.norm(q) for q in rng.normal(size=(50, 4))]
        quats += [np.array([0.0, 0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
                  np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])]
        for q in quats:
            m = quat_to_matrix(*q)
            np.testing.assert_allclose(quat_to_matrix(*matrix_to_quat(m)), m, atol=1e-9)


FINGERS = dict(finger_tip=0.005, finger_thick=0.025, taper_height=0.030, finger_max=0.030,
               finger_width=0.015, finger_width_root=0.025, finger_length=0.100,
               known_short=0.0285, known_long=0.034, floor_clearance=0.008, close_drop=0.009)
STANDING = (0.02, 0.01, 0.034, 0.0285, 0.0, 0.050)


class TestRoll(unittest.TestCase):

    def corner_scene(self, gap=0.016):
        """Standing box with walls (75 mm) past its +a end and +b side."""
        scene = Scene()
        scene.raise_box(*STANDING)
        x, y = STANDING[:2]
        scene.raise_box(x + 0.017 + gap + 0.005, y, 0.010, 0.120, 0.0, 0.075)
        scene.raise_box(x, y + 0.01425 + gap + 0.005, 0.120, 0.010, 0.0, 0.075)
        return scene, scene.target(*STANDING, period=math.pi / 2, pose='standing')

    def test_walls_block_every_upright_grasp(self):
        scene, target = self.corner_scene()
        g = scene.plan(target, params=replace(PARAMS, **FINGERS))
        self.assertIsNone(g.chosen)
        self.assertTrue(all(c.reason.startswith('blocked') for c in g.candidates))

    def test_roll_leans_the_thick_finger_top_off_the_wall(self):
        scene, target = self.corner_scene()
        params = replace(PARAMS, max_roll_deg=25.0, **FINGERS)
        g = scene.plan(target, params=params)
        c = g.chosen
        self.assertIsNotNone(c)
        self.assertTrue(0 < abs(c.roll) <= math.radians(25) + 1e-9)
        self.assertTrue(all(abs(o.roll) > 0 for o in g.options))
        np.testing.assert_allclose(g.rotation.T @ g.rotation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(g.rotation), 1.0)
        # closing axis rises by the roll; both fingertips stay between the floor and the top
        self.assertAlmostEqual(abs(g.rotation[2, 1]), math.sin(abs(g.options[0].roll))
                               * math.cos(g.options[0].tilt), delta=1e-6)
        rise = c.opening / 2 * math.sin(abs(c.roll)) * math.cos(c.tilt)
        self.assertLessEqual(c.tip_height + rise, STANDING[5] - params.roll_top_margin + 1e-9)
        self.assertGreaterEqual(c.tip_height - rise, params.floor_clearance + params.close_drop
                                * math.cos(c.tilt) - 1e-9)

    def test_free_object_is_never_rolled(self):
        scene = Scene()
        scene.raise_box(*STANDING)
        g = scene.plan(scene.target(*STANDING, period=math.pi / 2, pose='standing'),
                       params=replace(PARAMS, max_roll_deg=25.0, **FINGERS))
        self.assertEqual(g.chosen.roll, 0.0)
        self.assertTrue(all(o.roll == 0.0 for o in g.options))

    def test_too_short_object_cannot_roll(self):
        # 20 mm high: the raised fingertip would leave the side face at any roll step
        scene = Scene()
        low = STANDING[:5] + (0.020,)
        scene.raise_box(*low)
        x, y = low[:2]
        scene.raise_box(x + 0.017 + 0.021, y, 0.010, 0.120, 0.0, 0.075)
        scene.raise_box(x, y + 0.01425 + 0.021, 0.120, 0.010, 0.0, 0.075)
        g = scene.plan(scene.target(*low), params=replace(PARAMS, max_roll_deg=25.0, **FINGERS))
        self.assertIsNone(g.chosen)

    def test_roll_backups_follow_the_upright_options(self):
        scene = Scene()
        scene.raise_box(*STANDING)
        target = scene.target(*STANDING, period=math.pi / 2, pose='standing')
        params = replace(PARAMS, max_roll_deg=25.0, **FINGERS)
        plain = scene.plan(target, params=params)
        g = scene.plan(target, params=replace(params, roll_backup=True, roll_backup_max=4))
        n = len(plain.options)
        self.assertEqual([(o.name, o.tilt, o.roll) for o in g.options[:n]],
                         [(o.name, o.tilt, o.roll) for o in plain.options])
        np.testing.assert_allclose(g.position, plain.position)
        rolled = g.options[n:]
        self.assertTrue(rolled and all(o.roll for o in rolled))
        self.assertLessEqual(len(rolled), 4 * len(g.candidates))

    def test_narrow_opening_when_the_normal_one_is_blocked(self):
        scene = Scene()
        scene.raise_box(*STANDING)
        x, y = STANDING[:2]
        scene.raise_box(x + 0.017 + 0.040, y, 0.010, 0.120, 0.0, 0.075)
        scene.raise_box(x, y + 0.01425 + 0.040, 0.120, 0.010, 0.0, 0.075)
        target = scene.target(*STANDING, period=math.pi / 2, pose='standing')
        params = replace(PARAMS, **FINGERS)
        self.assertIsNone(scene.plan(target, params=params).chosen)
        g = scene.plan(target, params=replace(params, backup_clearances=(0.008, 0.005)))
        self.assertTrue(g.chosen.backup)
        self.assertAlmostEqual(g.chosen.opening, g.chosen.grip + 2 * 0.005)
        # the measured top stays the object's, not a backup opening
        self.assertAlmostEqual(g.length_m, STANDING[2], delta=0.003)
        self.assertAlmostEqual(g.width_m, STANDING[3], delta=0.003)
        self.assertTrue(all(o.backup and o.opening < 0.034 + 2 * 0.012 for o in g.options))

    def test_backup_order_leans_away_from_the_nearer_wall_first(self):
        free = {(0.3, 0.1), (0.3, -0.1), (0.2, 0.1), (0.1, 0.2), (0.1, -0.2)}
        rolls, tilts = (0.1, 0.2, 0.3), (0.3, 0.2, 0.1)
        self.assertEqual(backup_rolls(rolls, tilts, -1, 4, lambda t, r: (t, r) in free),
                         [(0.3, -0.1), (0.1, -0.2), (0.3, 0.1), (0.1, 0.2)])
        self.assertEqual(backup_rolls(rolls, tilts, 1, 2, lambda t, r: (t, r) in free),
                         [(0.1, 0.2), (0.1, -0.2)])
        # object 0.1 m along +x from the center of a 0.32 x 0.24 basket: + roll leans to -x
        n, x = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
        basket = Basket(np.zeros(3), x, 0.32, 0.24, '')
        self.assertEqual(away_roll_sign(0.1 * x, x, n, basket, BASKET), 1)
        self.assertEqual(away_roll_sign(-0.1 * x, x, n, basket, BASKET), -1)
        self.assertEqual(away_roll_sign(0.1 * x, x, n, None, BASKET), 0)


if __name__ == '__main__':
    unittest.main()
