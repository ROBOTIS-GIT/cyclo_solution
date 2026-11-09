"""Tool-frame/geometry regression checks; no ROS graph or robot commands."""
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'scripts'))
sys.path.insert(0, str(SOURCE / 'modules/dynamixel_pick/ffw_pick'))
from generate_sg2_gripper_model import extended_model
from ffw_pick.kinematics import Chain, tip
from ffw_pick.analytic_ik import ArmIK


class TestExtendedTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(get_package_share_directory('ffw_description')) / 'urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf'
        cls.original = base.read_text()
        cls.extended = extended_model(cls.original)

    def test_original_frames_are_preserved_and_new_tcp_is_exactly_45mm(self):
        for side in ('l', 'r'):
            old = Chain(self.original, 'base_link', f'end_effector_{side}_link')
            same = Chain(self.extended, 'base_link', f'end_effector_{side}_link')
            new = Chain(self.extended, 'base_link', f'gripper_{side}_tcp')
            values = {f'arm_{side}_joint{i}': q for i, q in enumerate(
                (1.05, 0.2 if side == 'l' else -0.2, 0.1, -2.27, 0.3, 0.35, 0.2), 1)}
            np.testing.assert_allclose(same.fk(values), old.fk(values), atol=1e-12)
            # New TCP + zero calibration equals the old explicit 45 mm tip offset.
            np.testing.assert_allclose(tip(new, values, 0), tip(old, values, .045), atol=1e-12)
            np.testing.assert_allclose(new.fk(values)[:3, :3], old.fk(values)[:3, :3])
            arm = ArmIK(new, [f'arm_{side}_joint{i}' for i in range(1, 8)])
            pose = new.fk(values)
            rows, _ = arm.solve(values, pose[:3, 3], pose[:3, :3], q7=values[f'arm_{side}_joint7'])
            self.assertTrue(len(rows))
            expected = np.array([values[n] for n in arm.names])
            self.assertLess(np.abs(rows - expected).max(axis=1).min(), 1e-6)

    def test_extension_collisions_follow_each_distal_finger(self):
        root = ET.fromstring(self.extended)
        for side in ('l', 'r'):
            for finger in ('r2', 'l2'):
                link = root.find(f"link[@name='gripper_{side}_rh_p12_rn_{finger}']")
                boxes = [part for part in link.findall('collision')
                         if part.get('name', '').startswith('demo_extension_')]
                self.assertEqual(len(boxes), 5)
                ends = []
                for part in boxes:
                    center = float(part.find('origin').get('xyz').split()[2])
                    height = float(part.find('geometry/box').get('size').split()[2])
                    ends.append(center + height / 2)
                self.assertAlmostEqual(max(ends), .040 + .045, places=8)
                self.assertIsNotNone(link.find('collision/geometry/mesh'))

    def test_cannot_apply_extension_twice(self):
        with self.assertRaisesRegex(ValueError, 'already'):
            extended_model(self.extended)


if __name__ == '__main__':
    unittest.main()
