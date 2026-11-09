import os

import cv2
from ffw_vision.camera_config import declare_camera_parameters, load_camera_config
from ffw_vision.depth_stats import (
    clip_roi, color_from_msg, depth_from_msg, roi_summary, valid_mask)
from ffw_vision.input_check import NS_PER_MS, NS_PER_SEC, stamp_to_ns
from ffw_vision.overlay import depth_colormap, draw_grid, draw_roi
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image

BUFFER_SIZE = 30


class DepthProbeNode(Node):

    def __init__(self):
        super().__init__('depth_probe')
        declare_camera_parameters(self)
        self.declare_parameter('roi', [0, 0, 0, 0])  # u_min, v_min, u_max, v_max
        self.declare_parameter('frames', 30)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('depth_scale', 0.001)  # meters per 16UC1 unit
        self.declare_parameter('output_dir', '/workspace/vision_ws/captures')
        self.declare_parameter('timeout_sec', 10.0)
        self.config = load_camera_config(self)
        if not self.config.depth_topic:
            raise ValueError(f'camera {self.config.camera_id} has no depth_topic')

        self.frames = self.get_parameter('frames').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.deadline_ns = self.now_ns() + int(
            self.get_parameter('timeout_sec').value * NS_PER_SEC)
        self.images = {}
        self.depths = {}
        self.received = {'image': 0, 'depth': 0}
        self.depth_frames = []
        self.last_color = None
        self.done = False
        self.exit_code = 0

        self.create_subscription(
            Image, self.config.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.config.depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_timer(0.5, self.check_timeout)
        self.get_logger().info(f'collecting {self.frames} frames from {self.config}')

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def on_image(self, msg):
        self.received['image'] += 1
        self.store(self.images, msg)

    def on_depth(self, msg):
        self.received['depth'] += 1
        self.store(self.depths, msg)

    def store(self, buffer, msg):
        if self.done:
            return
        stamp = stamp_to_ns(msg.header.stamp)
        buffer[stamp] = msg
        if len(buffer) > BUFFER_SIZE:
            del buffer[min(buffer)]
        if stamp in self.images and stamp in self.depths:
            self.last_color = self.images.pop(stamp)
            self.depth_frames.append(depth_from_msg(self.depths.pop(stamp), self.depth_scale))
            if len(self.depth_frames) >= self.frames:
                self.finish()

    def check_timeout(self):
        if self.done or self.now_ns() < self.deadline_ns:
            return
        gap = '-'
        if self.images and self.depths:
            gap = f'{min(abs(a - b) for a in self.images for b in self.depths) / NS_PER_MS:.1f}ms'
        self.get_logger().error(
            f'timeout: pairs={len(self.depth_frames)}/{self.frames} '
            f"images={self.received['image']} depths={self.received['depth']} "
            f'nearest_unpaired_gap={gap}')
        self.stop(1)

    def finish(self):
        try:
            self.write_results()
            self.stop(0)
        except (ValueError, OSError, cv2.error) as error:
            self.get_logger().error(str(error))
            self.stop(1)

    def write_results(self):
        height, width = self.depth_frames[0].shape
        roi = clip_roi(tuple(self.get_parameter('roi').value), width, height)
        summary = roi_summary(self.depth_frames, roi, self.min_depth, self.max_depth)

        color = color_from_msg(self.last_color)
        depth = self.depth_frames[-1]
        depth_vis, low, high = depth_colormap(
            depth, valid_mask(depth, self.min_depth, self.max_depth))
        blend = cv2.addWeighted(color, 0.5, depth_vis, 0.5, 0)

        output_dir = self.get_parameter('output_dir').value
        os.makedirs(output_dir, exist_ok=True)
        stamp = stamp_to_ns(self.last_color.header.stamp)
        prefix = os.path.join(output_dir, f'{stamp}_{self.config.camera_id}')
        paths = []
        for name, image in (('color', color), ('depth', depth_vis), ('overlay', blend)):
            path = f'{prefix}_{name}.png'
            if not cv2.imwrite(path, draw_roi(draw_grid(image), roi)):
                raise OSError(f'failed to write {path}')
            paths.append(path)

        self.get_logger().info(
            f"roi={roi} frames={summary['frames']} pixels={summary['pixels']} "
            f"valid_ratio={summary['valid_ratio']:.3f} "
            f"(min {summary['valid_ratio_min']:.3f}, max {summary['valid_ratio_max']:.3f}) "
            f"median={summary['median_m']:.3f}m p10={summary['p10_m']:.3f}m "
            f"p90={summary['p90_m']:.3f}m median_std={summary['median_std_m'] * 1000:.1f}mm "
            f"| images={self.received['image']} depths={self.received['depth']} "
            f'pairs={len(self.depth_frames)} | colormap {low:.2f}-{high:.2f}m')
        for path in paths:
            self.get_logger().info(f'saved {path}')

    def stop(self, code):
        self.exit_code = code
        self.done = True


def main(args=None):
    # Ctrl+C raises KeyboardInterrupt only; shutdown happens once below
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = DepthProbeNode()
    except ValueError as error:
        get_logger('depth_probe').error(str(error))
        rclpy.try_shutdown()
        return 1
    code = 1
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = node.exit_code
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return code
