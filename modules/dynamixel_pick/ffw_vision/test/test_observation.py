import math
import unittest

from ffw_vision.blob_detector import Blob
from ffw_vision.camera_config import CameraConfig
from ffw_vision.frame_registry import FrameInfo
from ffw_vision.observation import make_object_array, make_selected_observation
from ffw_vision.seg_detector import SegObject
from ffw_vision_msgs.msg import ObjectObservation, ObjectObservationArray, PixelObservation
import numpy as np
from std_msgs.msg import Header

CONFIG = CameraConfig('head', '/zed/zed_node/left/image_rect_color', 'rectified',
                      '/zed/zed_node/left/camera_info', '', 'zed_left_camera_optical_frame')
FRAME = FrameInfo(1792740000, 123456789, 'zed_left_camera_optical_frame', 672, 376, 0)


class TestSelectedObservation(unittest.TestCase):

    def test_fields_come_from_the_selected_frame(self):
        msg = make_selected_observation(CONFIG, FRAME, 10.0, 20.0, 'basket')
        self.assertEqual((msg.header.stamp.sec, msg.header.stamp.nanosec),
                         (1792740000, 123456789))
        self.assertEqual(msg.header.frame_id, 'zed_left_camera_optical_frame')
        self.assertEqual((msg.camera_id, msg.image_topic),
                         ('head', '/zed/zed_node/left/image_rect_color'))
        self.assertEqual((msg.image_width, msg.image_height), (672, 376))
        self.assertEqual(msg.point_type, 'selected_point')
        self.assertEqual(msg.source_mode, PixelObservation.SOURCE_MANUAL)
        self.assertEqual(msg.status, PixelObservation.STATUS_VALID)
        self.assertEqual((msg.u, msg.v, msg.target_label), (10.0, 20.0, 'basket'))

    def test_corner_pixels_are_valid(self):
        make_selected_observation(CONFIG, FRAME, 0.0, 0.0)
        make_selected_observation(CONFIG, FRAME, 671.0, 375.0)

    def test_rejects_outside_or_non_finite(self):
        for u, v in ((672.0, 0.0), (0.0, 376.0), (-1.0, 0.0), (float('nan'), 0.0),
                     (0.0, float('inf'))):
            with self.subTest(u=u, v=v), self.assertRaises(ValueError):
                make_selected_observation(CONFIG, FRAME, u, v)


class TestObjectArray(unittest.TestCase):

    def setUp(self):
        self.header = Header(frame_id='zed_left_camera_optical_frame')
        self.header.stamp.sec, self.header.stamp.nanosec = 1792740000, 123456789

    def test_empty_is_not_detected(self):
        msg = make_object_array(CONFIG, self.header, 672, 376, [], 'dynamixel')
        self.assertEqual(msg.status, ObjectObservationArray.STATUS_NOT_DETECTED)
        self.assertEqual(msg.source_mode, ObjectObservationArray.SOURCE_AUTO)
        self.assertEqual(list(msg.objects), [])

    def test_fields_and_header(self):
        blobs = [Blob(10.0, 20.0, 44.0, 32.0, 0.5, math.pi, 1400, 0.03, 'lying', False),
                 Blob(30.0, 40.0, 80.0, 46.0, -0.2, 0.0, 2300, float('nan'), 'unknown', True)]
        msg = make_object_array(CONFIG, self.header, 672, 376, blobs, 'dynamixel')
        self.assertEqual(msg.header, self.header)
        self.assertEqual((msg.camera_id, msg.image_width, msg.image_height),
                         ('head', 672, 376))
        self.assertEqual(msg.status, ObjectObservationArray.STATUS_VALID)
        first, second = msg.objects
        self.assertEqual((first.target_label, first.pose_label, first.status),
                         ('dynamixel', 'lying', ObjectObservation.STATUS_VALID))
        self.assertEqual((first.u, first.v, first.length, first.width, first.angle),
                         (10.0, 20.0, 44.0, 32.0, 0.5))
        self.assertEqual(first.angle_period, math.pi)
        self.assertFalse(first.has_score)
        self.assertTrue(math.isnan(first.top_height))
        self.assertFalse(msg.has_support_plane)
        self.assertEqual(second.status, ObjectObservation.STATUS_AMBIGUOUS)

    def test_seg_objects_carry_score_height_and_plane(self):
        obj = SegObject(250.0, 105.0, 30.0, 26.0, 0.3, math.pi / 2, 'standing', False, 0.84,
                        0.049, 0.027, 0.025, np.zeros((4, 1, 2), np.int32))
        plane = (np.array([0.0, -0.7, -0.7]), 0.262)
        msg = make_object_array(CONFIG, self.header, 424, 240, [obj], 'dynamixel', plane)
        self.assertTrue(msg.has_support_plane)
        self.assertEqual(list(msg.support_plane), [0.0, -0.7, -0.7, 0.262])
        (out,) = msg.objects
        self.assertTrue(out.has_score)
        self.assertAlmostEqual(out.score, 0.84, places=5)
        self.assertAlmostEqual(out.top_height, 0.049)
        self.assertEqual((out.u, out.v, out.pose_label), (250.0, 105.0, 'standing'))


if __name__ == '__main__':
    unittest.main()
