import unittest

from ffw_vision.camera_config import CameraConfig, make_camera_config
from ffw_vision.input_check import InputMonitor, NS_PER_MS, NS_PER_SEC

HEAD = {
    'image_topic': '/zed/zed_node/left/image_rect_color',
    'stream_type': 'rectified',
    'camera_info_topic': '/zed/zed_node/left/camera_info',
    'depth_topic': '',
    'frame_id': 'zed_left_camera_optical_frame',
}
FRAME = HEAD['frame_id']
T0 = 10 * NS_PER_SEC
MS = NS_PER_MS


def make_monitor(depth_topic=''):
    config = CameraConfig('head', **{**HEAD, 'depth_topic': depth_topic})
    return InputMonitor(config, timeout_ns=NS_PER_SEC)


def add_frame(monitor, stamp, rx, info=True, depth_encoding=None):
    if info:
        monitor.on_camera_info(stamp, FRAME, 672, 376)
    monitor.on_image(stamp, FRAME, 672, 376, 'bgra8', rx)
    if depth_encoding:
        monitor.on_depth(stamp, FRAME, 672, 376, depth_encoding)


class TestCameraConfig(unittest.TestCase):

    def test_valid(self):
        config = make_camera_config('head', HEAD)
        self.assertEqual(config.camera_id, 'head')
        self.assertEqual(config.depth_topic, '')

    def test_rejects_empty_field(self):
        for field in ('image_topic', 'stream_type', 'camera_info_topic', 'frame_id'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                make_camera_config('head', {**HEAD, field: ''})

    def test_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            make_camera_config('head', {**HEAD, 'stream_type': 'depth'})
        with self.assertRaises(ValueError):
            make_camera_config('tail', HEAD)


class TestInputMonitor(unittest.TestCase):

    def test_paired_frames_ok(self):
        monitor = make_monitor()
        for i in range(3):
            add_frame(monitor, T0 + i * 33 * MS, T0 + i * 33 * MS + 5 * MS)
        result = monitor.report(T0 + 300 * MS)
        self.assertEqual(result['state'], 'OK')
        self.assertEqual(result['info'], (3, 3, 0, 0))
        self.assertIsNone(result['depth'])
        self.assertAlmostEqual(result['max_interval_ms'], 33.0)
        self.assertAlmostEqual(result['latency_max_ms'], 5.0)

    def test_late_camera_info_is_still_paired(self):
        monitor = make_monitor()
        add_frame(monitor, T0, T0)
        add_frame(monitor, T0 + 33 * MS, T0 + 33 * MS, info=False)
        first = monitor.report(T0 + 120 * MS)
        self.assertEqual((first['info'][1:3], first['waiting']), ((1, 0), 1))

        monitor.on_camera_info(T0 + 33 * MS, FRAME, 672, 376)
        second = monitor.report(T0 + 200 * MS)
        self.assertEqual((second['info'][1:3], second['waiting']), ((1, 0), 0))

    def test_unpaired_image_reports_gap(self):
        monitor = make_monitor()
        monitor.on_camera_info(T0, FRAME, 672, 376)
        monitor.on_camera_info(T0 + 40 * MS, FRAME, 672, 376)
        monitor.on_image(T0 + 10 * MS, FRAME, 672, 376, 'bgra8', T0 + 20 * MS)
        result = monitor.report(T0 + 200 * MS)
        self.assertEqual(result['state'], 'MISMATCH')
        self.assertEqual(result['info'], (2, 0, 1, 10 * MS))

    def test_frame_id_and_size_mismatch(self):
        monitor = make_monitor()
        monitor.on_camera_info(T0, FRAME, 672, 376)
        monitor.on_image(T0, 'other_frame', 1344, 376, 'bgra8', T0)
        result = monitor.report(T0)
        self.assertEqual(result['state'], 'MISMATCH')
        self.assertEqual(result['errors'], {'frame_id': 1, 'size': 1})

    def test_missing_camera_info(self):
        monitor = make_monitor()
        monitor.on_image(0, FRAME, 672, 376, 'bgra8', 0)
        self.assertEqual(monitor.report(10 * MS)['state'], 'NO_CAMERA_INFO')

    def test_no_input_after_timeout(self):
        monitor = make_monitor()
        self.assertEqual(monitor.report(0)['state'], 'NO_INPUT')
        monitor.on_image(0, FRAME, 672, 376, 'bgra8', 0)
        self.assertEqual(monitor.report(2 * NS_PER_SEC)['state'], 'NO_INPUT')


class TestDepth(unittest.TestCase):

    def test_depth_paired(self):
        monitor = make_monitor('/zed/zed_node/depth/depth_registered')
        for i in range(2):
            add_frame(monitor, T0 + i * 33 * MS, T0 + i * 33 * MS, depth_encoding='32FC1')
        result = monitor.report(T0 + 300 * MS)
        self.assertEqual(result['state'], 'OK')
        self.assertEqual(result['depth'], (2, 2, 0, 0))

    def test_missing_depth(self):
        monitor = make_monitor('/zed/zed_node/depth/depth_registered')
        add_frame(monitor, T0, T0)
        self.assertEqual(monitor.report(T0 + 200 * MS)['state'], 'NO_DEPTH')

    def test_depth_encoding_mismatch(self):
        monitor = make_monitor('/zed/zed_node/depth/depth_registered')
        add_frame(monitor, T0, T0, depth_encoding='8UC1')
        result = monitor.report(T0 + 200 * MS)
        self.assertEqual(result['state'], 'MISMATCH')
        self.assertEqual(result['errors'], {'depth_encoding': 1})


if __name__ == '__main__':
    unittest.main()
