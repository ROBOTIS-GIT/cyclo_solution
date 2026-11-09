"""Action lifecycle with a fake engine; no controllers or cameras are contacted."""
import threading
import time
import unittest
from unittest.mock import patch

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from ffw_vision_msgs.action import PickDynamixel
from ffw_pick.action_server import PickServer


class TestPickAction(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.server = PickServer()
        self.client_node = rclpy.create_node('test_pick_client')
        self.client = ActionClient(self.client_node, PickDynamixel, '/dynamixel_pick/execute')
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.server)
        self.executor.add_node(self.client_node)
        self.thread = threading.Thread(target=self.executor.spin)
        self.thread.start()
        self.assertTrue(self.client.wait_for_server(timeout_sec=3))

    def tearDown(self):
        self.server.stopping.set()
        self.executor.shutdown()
        self.thread.join(3)
        self.client_node.destroy_node()
        self.server.destroy_node()
        rclpy.shutdown()

    def wait(self, future):
        deadline = time.monotonic() + 3
        while not future.done() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(future.done())
        return future.result()

    def send(self):
        return self.wait(self.client.send_goal_async(
            PickDynamixel.Goal(execute=True, move_to_observe=True)))

    def test_success_does_not_require_or_claim_holding(self):
        def engine(args, hooks):
            self.assertIn('--run', args)
            self.assertIn('--no-finger-boxes', args)
            hooks.motion_started = True
            return 0
        with patch('ffw_pick.pick_moveit.main', engine):
            goal = self.send()
            result = self.wait(goal.get_result_async())
            self.assertEqual(result.status, 4)
            self.assertFalse(result.result.holding)
            self.assertIn("not verified", result.result.message)
            self.assertFalse(result.result.retryable)

    def test_perception_failure_can_retry_but_execution_failure_cannot(self):
        for moved in (False, True):
            def engine(args, hooks):
                hooks.motion_started = moved
                hooks.error_code = 'OBJECT_NOT_FOUND'
                return 1
            with patch('ffw_pick.pick_moveit.main', engine):
                goal = self.send()
                result = self.wait(goal.get_result_async()).result
                self.assertEqual(result.retryable, not moved)

    def test_cancel_and_single_goal_arbitration(self):
        entered = threading.Event()
        def engine(args, hooks):
            entered.set()
            while True:
                try:
                    hooks.check_cancel()
                except KeyboardInterrupt:
                    hooks.error_code = 'CANCELLED'
                    return 1
                time.sleep(.01)
        with patch('ffw_pick.pick_moveit.main', engine):
            goal = self.send()
            self.assertTrue(entered.wait(2))
            other = self.send()
            self.assertFalse(other.accepted)
            self.wait(goal.cancel_goal_async())
            result = self.wait(goal.get_result_async())
            self.assertEqual(result.status, 5)
            self.assertTrue(result.result.stop_confirmed)

    def test_unknown_scene_latches_backend(self):
        def engine(args, hooks):
            hooks.error_code = 'SCENE_UNCONFIRMED'
            return 1
        with patch('ffw_pick.pick_moveit.main', engine):
            goal = self.send()
            result = self.wait(goal.get_result_async()).result
            self.assertFalse(result.retryable)
            self.assertFalse(self.send().accepted)

    def test_unknown_stop_latches_backend(self):
        def engine(args, hooks):
            hooks.motion_started = True
            hooks.stop_confirmed = False
            raise RuntimeError('lost controller result')
        with patch('ffw_pick.pick_moveit.main', engine):
            goal = self.send()
            result = self.wait(goal.get_result_async()).result
            self.assertEqual(result.error_code, 'STOP_UNCONFIRMED')
            self.assertFalse(self.send().accepted)


if __name__ == '__main__':
    unittest.main()
