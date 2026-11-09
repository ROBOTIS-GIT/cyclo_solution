import os
import unittest

from ffw_pick.config import ArmConfig, from_file, PickConfig, require, ros_parameters
from ffw_pick.gripper import load_gripper
from ffw_pick.kinematics import Chain
from ffw_pick.pick_moveit import TASK_KEYS

HERE = os.path.dirname(__file__)
CONFIG = os.path.join(HERE, '..', 'config')
GRASP_CONFIG = os.path.join(HERE, '..', '..', 'ffw_grasp', 'config')


class TestInstalledSettings(unittest.TestCase):
    """The config files hold what pick_moveit had in its code before (2026-11-02)."""

    def test_arm(self):
        arm = from_file(ArmConfig, os.path.join(CONFIG, 'arm_r.yaml'))
        self.assertEqual(arm.group, 'arm_r')
        self.assertEqual(arm.base_frame, 'base_link')
        self.assertEqual(arm.end_effector_link, 'gripper_r_tcp')
        self.assertEqual(arm.wrist_link, 'arm_r_link7')
        self.assertEqual(arm.arm_joints, tuple(f'arm_r_joint{i}' for i in range(1, 8)))
        self.assertEqual(arm.gripper_joint, 'gripper_r_joint1')
        self.assertEqual(arm.finger_links, ('gripper_r_rh_p12_rn_r1', 'gripper_r_rh_p12_rn_r2',
                                            'gripper_r_rh_p12_rn_l1', 'gripper_r_rh_p12_rn_l2'))
        self.assertEqual(arm.touch_links, (
            'end_effector_r_link', 'gripper_r_rh_p12_rn_base', 'gripper_r_rh_p12_rn_r1',
            'gripper_r_rh_p12_rn_r2', 'gripper_r_rh_p12_rn_l1', 'gripper_r_rh_p12_rn_l2'))
        self.assertEqual(arm.gripper_action, '/arm_r_controller/follow_joint_trajectory')
        self.assertEqual(arm.controller_topic,
                         '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory')
        self.assertEqual(arm.home, (1.05, 0.0, 0.0, -2.27, 0.0, 0.35, 0.0))
        self.assertEqual((arm.home_tolerance, arm.joint_margin), (0.05, 0.003))
        self.assertEqual((arm.max_acceleration, arm.max_fingertip_speed), (5.0, 1.0))

    def test_joint_speed_from_the_urdf(self):
        arm = from_file(ArmConfig, os.path.join(CONFIG, 'arm_r.yaml'))
        with open(os.path.join(HERE, 'arm_r.urdf')) as f:
            chain = Chain(f.read(), arm.base_frame, arm.end_effector_link)
        self.assertEqual(min(chain.velocity(n) for n in arm.arm_joints), 4.8)

    def test_pick(self):
        pick = from_file(PickConfig, os.path.join(CONFIG, 'pick.yaml'))
        self.assertEqual(pick, PickConfig(
            ns='/ffw_grasp/right_wrist', above=0.10, lift=0.15, speed=0.35, accel=0.35,
            turn_window=15.0, budget=30.0, hold=2.0, home_lift=0.10, plan_tries=3,
            retry_wait=2.0, empty_tries=3, state_tolerance=0.01))
        self.assertIsInstance(pick.plan_tries, int)
        self.assertIsInstance(pick.turn_window, float)

    def test_gripper(self):
        gripper = load_gripper(os.path.join(CONFIG, 'gripper_table.yaml'))
        self.assertEqual(gripper.table, [(0.0, 0.092), (0.2, 0.077), (0.4, 0.061),
                                         (0.6, 0.043), (0.8, 0.022)])
        self.assertEqual((gripper.hold_below, gripper.close_command, gripper.move_time),
                         (0.94, 1.0, 1.0))

    def test_grasp_task_and_calibration(self):
        task_file = os.path.join(GRASP_CONFIG, 'dynamixel_top_grasp.yaml')
        calibration_file = os.path.join(GRASP_CONFIG, 'robot_calibration.yaml')
        if not os.path.exists(task_file):
            self.skipTest('ffw_grasp is not next to ffw_pick')
        task = {**ros_parameters(task_file), **ros_parameters(calibration_file)}
        require(task, TASK_KEYS, task_file)
        self.assertEqual((task['table_margin'], task['table_thick']), (0.15, 0.04))
        self.assertEqual(task['finger_overlap'], 0.01)
        self.assertEqual(task['finger_layers'], [0.0, 0.01, 0.02, 0.03, 0.045])
        self.assertEqual(task['tip_below_end_effector'], 0.0)

    def test_missing_or_unknown_keys_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            require({'a': 1}, ('a', 'b'), 'x.yaml')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            from_file(PickConfig, os.path.join(CONFIG, 'arm_r.yaml'))
