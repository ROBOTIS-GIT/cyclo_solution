import unittest

from ffw_vision.display_mapping import display_to_source
from ffw_vision.frame_registry import FrameInfo, FrameRegistry

SOURCE = (672, 376)


class TestDisplayToSource(unittest.TestCase):

    def test_same_size(self):
        self.assertEqual(display_to_source(0, 0, SOURCE, SOURCE), (0.0, 0.0))
        self.assertEqual(display_to_source(10.9, 20.2, SOURCE, SOURCE), (10.0, 20.0))
        self.assertEqual(display_to_source(671.99, 375.99, SOURCE, SOURCE), (671.0, 375.0))

    def test_half_size(self):
        self.assertEqual(display_to_source(168, 94, (336, 188), SOURCE), (336.0, 188.0))
        self.assertEqual(display_to_source(335.9, 187.9, (336, 188), SOURCE), (671.0, 375.0))

    def test_non_integer_scale(self):
        u, v = display_to_source(500.0, 279.0, (1000.5, 559.8), SOURCE)
        self.assertEqual((u, v), (335.0, 187.0))

    def test_outside_or_invalid_display(self):
        for x, y in ((-0.1, 0), (0, -0.1), (672, 0), (0, 376)):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                display_to_source(x, y, SOURCE, SOURCE)
        with self.assertRaises(ValueError):
            display_to_source(0, 0, (0, 376), SOURCE)


class TestFrameRegistry(unittest.TestCase):

    def frame(self, sec, rx_sec):
        return FrameInfo(sec, 5, 'zed_left_camera_optical_frame', 672, 376, rx_sec * 10**9)

    def test_lookup_by_stamp(self):
        registry = FrameRegistry(retention_ns=10 * 10**9)
        registry.add(self.frame(100, 100))
        self.assertEqual(registry.get(100 * 10**9 + 5).frame_id, 'zed_left_camera_optical_frame')
        self.assertIsNone(registry.get(100 * 10**9))

    def test_old_frames_expire(self):
        registry = FrameRegistry(retention_ns=10 * 10**9)
        registry.add(self.frame(100, 100))
        registry.add(self.frame(105, 105))
        registry.add(self.frame(112, 112))
        self.assertIsNone(registry.get(100 * 10**9 + 5))
        self.assertIsNotNone(registry.get(105 * 10**9 + 5))
        self.assertIsNotNone(registry.get(112 * 10**9 + 5))


if __name__ == '__main__':
    unittest.main()
