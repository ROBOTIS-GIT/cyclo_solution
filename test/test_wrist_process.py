"""Bounded frame processing and timestamp contracts; no robot/camera required."""
from collections import deque
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wrist_perception import WristPerception


class ProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = WristPerception()
        self.node.buffer = Mock()
        self.node.buffer.lookup_transform.return_value = TransformStamped()
        self.node.detect = Mock()
        self.node.publish = Mock()
        info = CameraInfo()
        info.width, info.height = 320, 240
        info.header.frame_id = "camera_left_depth_optical_frame"
        info.p = [200., 0., 160., 0., 0., 200., 120., 0., 0., 0., 1., 0.]
        self.node.info = info

    def tearDown(self):
        self.node.destroy_node()

    def message(self, age):
        message = Image()
        message.width, message.height = 320, 240
        message.header.frame_id = self.node.info.header.frame_id
        message.header.stamp = (self.node.get_clock().now() - Duration(seconds=age)).to_msg()
        return message

    def stamp(self, message):
        return rclpy.time.Time.from_msg(message.header.stamp).nanoseconds

    def test_newest_ready_frame_only_and_no_replay(self):
        old, middle, newest = [self.message(age) for age in (.3, .2, .1)]
        self.node.pending.extend([old, middle, newest])
        self.node.process()
        self.node.detect.assert_called_once()
        self.assertIs(self.node.detect.call_args.args[0], newest)
        self.assertFalse(self.node.pending)
        self.node.process()
        self.node.detect.assert_called_once()

    def test_waiting_newest_tf_does_not_block_older_ready_frame(self):
        older, newest = self.message(.2), self.message(.1)
        def transform(target, source, stamp):
            self.assertNotEqual(stamp.nanoseconds, 0)  # Never silently substitute latest TF.
            if stamp.nanoseconds == self.stamp(newest):
                raise TransformException("future TF")
            return TransformStamped()
        self.node.buffer.lookup_transform.side_effect = transform
        self.node.pending.extend([older, newest])
        self.node.process()
        self.assertIs(self.node.detect.call_args.args[0], older)
        self.assertEqual(list(self.node.pending), [newest])
        self.node.publish.assert_not_called()
        self.node.buffer.lookup_transform.side_effect = None
        self.node.process()
        self.assertIs(self.node.detect.call_args.args[0], newest)
        self.assertEqual(self.node.detect.call_count, 2)

    def test_callback_returns_after_one_fit_even_if_input_arrives(self):
        first, later = self.message(.2), self.message(.1)
        self.node.detect.side_effect = lambda *_: self.node.pending.append(later)
        self.node.pending.append(first)
        self.node.process()
        self.node.detect.assert_called_once()
        self.assertEqual(list(self.node.pending), [later])

    def test_tf_timeout_invalidates_without_geometry_fit(self):
        self.node.buffer.lookup_transform.side_effect = TransformException("no TF")
        self.node.history.append((1, None))
        self.node.pending.append(self.message(.7))
        self.node.process()
        self.node.detect.assert_not_called()
        self.assertEqual(self.node.publish.call_args.args[1], "TF_UNAVAILABLE")
        self.assertFalse(self.node.history)

    def test_old_or_replayed_images_never_fit(self):
        replay = self.message(.1)
        self.node.last_stamp = self.stamp(replay)
        self.node.pending.extend([self.message(1.1), replay])
        self.node.process()
        self.node.detect.assert_not_called()

    def test_delayed_result_cannot_publish_valid_target(self):
        original = self.node.publish
        self.node.publish = WristPerception.publish.__get__(self.node)
        self.node.observations = Mock()
        self.node.status = Mock()
        self.node.history = deque([(1, None)], maxlen=3)
        self.node.publish(self.message(1.1), "VALID_RIM", fit={
            "center": [0., 0., .9], "error": .009, "coverage_deg": 101.})
        result = self.node.observations.publish.call_args.args[0]
        self.assertFalse(result.valid)
        self.assertTrue(result.detail.startswith("STALE_FRAME:"))
        self.assertFalse(self.node.history)
        self.node.publish = original


if __name__ == "__main__":
    unittest.main()
