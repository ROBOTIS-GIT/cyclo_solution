from ffw_vision.camera_config import declare_camera_parameters, load_camera_config
from ffw_vision.input_check import format_report, InputMonitor, NS_PER_SEC, stamp_to_ns
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image

SPIN_TIMEOUT_SEC = 0.1


class InputMonitorNode(Node):

    def __init__(self):
        super().__init__('input_monitor')
        declare_camera_parameters(self)
        self.declare_parameter('report_period_sec', 2.0)
        self.declare_parameter('timeout_sec', 1.0)
        self.config = load_camera_config(self)
        timeout_sec = self.get_parameter('timeout_sec').value
        self.monitor = InputMonitor(self.config, int(timeout_sec * NS_PER_SEC))

        self.create_subscription(
            Image, self.config.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.config.camera_info_topic, self.on_camera_info,
            qos_profile_sensor_data)
        if self.config.depth_topic:
            self.create_subscription(
                Image, self.config.depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_timer(self.get_parameter('report_period_sec').value, self.on_report)
        self.get_logger().info(f'monitoring {self.config}')

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def on_image(self, msg):
        self.monitor.on_image(
            stamp_to_ns(msg.header.stamp), msg.header.frame_id, msg.width, msg.height,
            msg.encoding, self.now_ns())

    def on_camera_info(self, msg):
        self.monitor.on_camera_info(
            stamp_to_ns(msg.header.stamp), msg.header.frame_id, msg.width, msg.height)

    def on_depth(self, msg):
        self.monitor.on_depth(
            stamp_to_ns(msg.header.stamp), msg.header.frame_id, msg.width, msg.height,
            msg.encoding)

    def on_report(self):
        result = self.monitor.report(self.now_ns())
        text = format_report(result)
        if result['state'] == 'OK':
            self.get_logger().info(text)
        else:
            self.get_logger().warning(text)


def main(args=None):
    # Ctrl+C raises KeyboardInterrupt only; shutdown happens once below
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = InputMonitorNode()
    except ValueError as error:
        get_logger('input_monitor').error(str(error))
        rclpy.try_shutdown()
        return 1
    try:
        # wait with a timeout: without signal handlers Ctrl+C is seen only between waits
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=SPIN_TIMEOUT_SEC)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
