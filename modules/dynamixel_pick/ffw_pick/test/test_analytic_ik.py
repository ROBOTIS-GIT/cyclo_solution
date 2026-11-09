import math
import os
import unittest

from ffw_pick.analytic_ik import ArmIK, Line, line_points, rot, shortest_path
from ffw_pick.config import ArmConfig, from_file
from ffw_pick.kinematics import Chain
import numpy as np

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, 'arm_r.urdf')) as f:
    URDF = f.read()
ARM_CFG = from_file(ArmConfig, os.path.join(HERE, '..', 'config', 'arm_r.yaml'))
ARM_JOINTS = list(ARM_CFG.arm_joints)
CHAIN = Chain(URDF, ARM_CFG.base_frame, ARM_CFG.end_effector_link)
ARM = ArmIK(CHAIN, ARM_JOINTS, margin=ARM_CFG.joint_margin)
HOME = list(ARM_CFG.home)
NOW = {'lift_joint': -0.1, **dict(zip(ARM_JOINTS, HOME))}
JUMP = 0.15


def joints(values):
    return np.array([values[n] for n in ARM_JOINTS])


def random_values(rng):
    return {'lift_joint': rng.uniform(-0.5, 0.0),
            **{n: rng.uniform(lo + 0.003, hi - 0.003)
               for n in ARM_JOINTS for lo, hi in [CHAIN.limits(n)]}}


def reaches(values, pose):
    got = CHAIN.fk(values)
    return (np.linalg.norm(got[:3, 3] - pose[:3, 3]) < 1e-6
            and np.abs(got[:3, :3] - pose[:3, :3]).max() < 1e-6)


def grasp_line(position, rotation):
    """Approach down the finger axis from 20 cm back, then 15 cm straight up."""
    return (line_points(position + 0.2 * rotation[:, 2], position, 0.005)
            + line_points(position, position + [0.0, 0.0, 0.15], 0.005)[1:])


class TestSolve(unittest.TestCase):

    def test_every_solution_reaches_and_the_true_one_is_found(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            values = random_values(rng)
            pose = CHAIN.fk(values)
            q, _ = ARM.solve(values, pose[:3, 3], pose[:3, :3], q7=values['arm_r_joint7'])
            self.assertTrue(len(q))
            self.assertLess(np.abs(q - joints(values)).max(axis=1).min(), 1e-6)
            for row in q:
                self.assertTrue(reaches({**values, **dict(zip(ARM_JOINTS, row))}, pose))

    def test_grid_has_a_solution_near_any_arm_shape(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            values = random_values(rng)
            pose = CHAIN.fk(values)
            q, branch = ARM.solve(values, pose[:3, 3], pose[:3, :3])
            self.assertEqual(len(q), len(branch))
            self.assertTrue(np.all((q >= ARM.lower) & (q <= ARM.upper)))
            self.assertLess(np.abs(q - joints(values)).max(axis=1).min(), 0.05)

    def test_out_of_reach_has_no_solution(self):
        q, _ = ARM.solve(NOW, [1.5, -0.3, 0.8], np.eye(3))
        self.assertEqual(len(q), 0)


class TestLine(unittest.TestCase):

    def test_grasp_and_lift_line_is_continuous_and_exact(self):
        rotation = rot(1, -0.4) @ rot(2, 0.7)
        points = grasp_line(np.array([0.42, -0.25, 0.80]), rotation)
        line = Line(ARM, NOW, points, rotation)
        path, why = line.path(joints(NOW), False, JUMP)
        self.assertEqual(why, '')
        self.assertEqual(len(path), len(points))
        self.assertLess(np.abs(np.diff(path, axis=0)).max(), JUMP)
        self.assertTrue(np.all((path >= ARM.lower) & (path <= ARM.upper)))
        for q, p in zip(path, points):
            got = CHAIN.fk({**NOW, **dict(zip(ARM_JOINTS, q))})
            self.assertLess(np.linalg.norm(got[:3, 3] - p), 1e-6)
            self.assertLess(np.abs(got[:3, :3] - rotation).max(), 1e-6)
        self.assertGreater(line.start_bound(joints(NOW)), 0.0)
        self.assertLessEqual(line.start_bound(joints(NOW)), np.abs(path[0] - joints(NOW)).max())

    def test_fixed_start_line_leaves_from_the_arm(self):
        start = {**NOW, 'arm_r_joint4': -1.9, 'arm_r_joint6': 0.6}
        pose = CHAIN.fk(start)
        points = line_points(pose[:3, 3], pose[:3, 3] + [0.0, 0.0, 0.05], 0.005)[1:]
        path, why = Line(ARM, start, points, pose[:3, :3]).path(joints(start), True, JUMP)
        self.assertEqual(why, '')
        self.assertLess(np.abs(path[0] - joints(start)).max(), JUMP)

    def test_unreachable_line_says_where(self):
        rotation = np.eye(3)
        points = line_points([0.45, -0.25, 0.80], [1.2, -0.25, 0.80], 0.01)
        line = Line(ARM, NOW, points, rotation)
        path, why = line.path(joints(NOW), False, JUMP)
        self.assertIsNone(path)
        self.assertRegex(why, r'no IK solution at \d+% of the line')

    def test_over_bound_is_not_filled(self):
        rotation = rot(1, -0.4) @ rot(2, 0.7)
        line = Line(ARM, NOW, grasp_line(np.array([0.42, -0.25, 0.80]), rotation), rotation)
        path, why = line.path(joints(NOW), False, JUMP, bound=0.1)
        self.assertIsNone(path)
        self.assertEqual(why, 'over bound')
        self.assertGreaterEqual(line.cost, 0.1)


class TestShortestPath(unittest.TestCase):

    def layers(self):
        # two ways through three points: small steps on one side, a cheap start on the other
        near = [np.full((1, 7), 0.10 * i) for i in range(3)]
        far = [np.full((1, 7), 1.0 + 0.01 * i) for i in range(3)]
        return [(np.vstack((a, b)), np.array([0, 1])) for a, b in zip(near, far)]

    def test_least_motion_wins_and_bans_are_respected(self):
        layers = self.layers()
        rows, cost, why = shortest_path(layers, np.full(7, 1.0), JUMP)
        self.assertEqual(rows, [1, 1, 1])  # start next to the far way
        self.assertAlmostEqual(cost, 0.02)
        banned = [np.array([False, True]), np.zeros(2, bool), np.zeros(2, bool)]
        rows, cost, why = shortest_path(layers, np.full(7, 1.0), JUMP, banned)
        self.assertEqual(rows, [0, 0, 0])
        self.assertAlmostEqual(cost, 1.0 + 0.2)

    def test_jump_is_not_a_path(self):
        layers = [(np.zeros((1, 7)), np.array([0])), (np.full((1, 7), 0.5), np.array([0]))]
        rows, cost, why = shortest_path(layers, np.zeros(7), JUMP)
        self.assertIsNone(rows)
        self.assertTrue(math.isinf(cost))
        self.assertIn('no continuous path', why)
