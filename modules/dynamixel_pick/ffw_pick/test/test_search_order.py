import math
import unittest

from ffw_pick.pick_moveit import plan_kind, search_tiers, summary_line
import numpy as np


def yaw(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestSearchTiers(unittest.TestCase):

    def test_normal_by_turn_then_backups_in_given_order(self):
        options = [{'rotation': yaw(40), 'backup': False}, {'rotation': yaw(10), 'backup': False},
                   {'rotation': yaw(5), 'backup': True}, {'rotation': yaw(60), 'backup': True}]
        upright, rolled = search_tiers(options, np.eye(3))
        turns = [e[0] for e in upright]
        self.assertEqual({e[1] for e in upright}, {0, 1})
        self.assertEqual(turns, sorted(turns))
        self.assertEqual([e[1] for e in upright][0], 1)
        # backups keep ffw_grasp's order, both yaws each
        self.assertEqual([(e[1], e[2]) for e in rolled], [(2, False), (2, True), (3, False),
                                                          (3, True)])
        # rows without the flag are normal
        self.assertEqual(len(search_tiers([{'rotation': yaw(0)}], np.eye(3))[0]), 2)


if __name__ == '__main__':
    unittest.main()


class TestRecovery(unittest.TestCase):

    def test_plan_kind(self):
        self.assertEqual(plan_kind('empty: 10 frames: no object in any frame'), 'empty')
        self.assertEqual(plan_kind('objects: 10 frames: 2 objects, none graspable: #1 ...'),
                         'objects')
        self.assertEqual(plan_kind("unsure: 10 frames: objects seen but none usable"), 'unsure')
        self.assertEqual(plan_kind('skipped: 10 frames: 1 objects, none graspable: #1 skipped'),
                         'skipped')
        # older ffw_grasp (no kind) and anything unknown: plan the same view again
        self.assertEqual(plan_kind('10 frames: only 4 of 4 frames agree'), 'view')
        self.assertEqual(plan_kind(''), 'view')

    def test_summary_line(self):
        line = summary_line({'cycle': 4, 'picked': 3, 'switch': 1}, 'basket empty')
        self.assertIn('4 cycles, 3 picked, 0 empty fingers, 1 other objects', line)
        self.assertTrue(line.endswith('ended: basket empty'))
