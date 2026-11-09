from dataclasses import replace

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from ffw_vision.camera_config import declare_camera_parameters, load_camera_config
from ffw_vision.frame_registry import FrameInfo, FrameRegistry
from ffw_vision.input_check import InputMonitor, NS_PER_SEC, stamp_to_ns
from ffw_vision.observation import make_selected_observation
from ffw_vision_msgs.msg import PixelObservation
from ffw_vision_msgs.srv import SelectPixel
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image

LEVELS = {
    'OK': DiagnosticStatus.OK,
    'NO_INPUT': DiagnosticStatus.ERROR,
}
SPIN_TIMEOUT_SEC = 0.1


class PixelObserverNode(Node):
    """Headless core: keeps frame headers, validates selections, publishes observations."""

    def __init__(self):
        super().__init__('pixel_observer')
        declare_camera_parameters(self)
        self.declare_parameter('retention_sec', 120.0)
        self.declare_parameter('status_period_sec', 1.0)
        self.declare_parameter('timeout_sec', 1.0)
        self.config = load_camera_config(self)
        self.registry = FrameRegistry(int(self.get_parameter('retention_sec').value * NS_PER_SEC))
        timeout_ns = int(self.get_parameter('timeout_sec').value * NS_PER_SEC)
        # depth is checked by the coordinate node, not here
        self.monitor = InputMonitor(replace(self.config, depth_topic=''), timeout_ns)

        self.observation_pub = self.create_publisher(PixelObservation, 'pixel_observation', 10)
        self.status_pub = self.create_publisher(DiagnosticArray, 'status', 10)
        self.create_service(SelectPixel, 'select_pixel', self.on_select)
        self.create_subscription(
            Image, self.config.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.config.camera_info_topic, self.on_camera_info,
            qos_profile_sensor_data)
        self.create_timer(self.get_parameter('status_period_sec').value, self.publish_status)
        self.get_logger().info(f'observing {self.config}')

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def on_image(self, msg):
        now = self.now_ns()
        self.registry.add(FrameInfo(msg.header.stamp.sec, msg.header.stamp.nanosec,
                                    msg.header.frame_id, msg.width, msg.height, now))
        self.monitor.on_image(stamp_to_ns(msg.header.stamp), msg.header.frame_id,
                              msg.width, msg.height, msg.encoding, now)

    def on_camera_info(self, msg):
        self.monitor.on_camera_info(
            stamp_to_ns(msg.header.stamp), msg.header.frame_id, msg.width, msg.height)

    def on_select(self, request, response):
        frame = self.registry.get(stamp_to_ns(request.image_stamp))
        try:
            if request.camera_id != self.config.camera_id:
                raise ValueError(
                    f'camera_id {request.camera_id!r} != {self.config.camera_id!r}')
            if frame is None:
                raise ValueError('frame not found: expired or never received')
            msg = make_selected_observation(
                self.config, frame, request.u, request.v, request.target_label)
        except ValueError as error:
            response.accepted = False
            response.message = str(error)
            self.get_logger().warning(f'rejected selection: {error}')
            return response
        self.observation_pub.publish(msg)
        response.accepted = True
        response.message = 'published'
        response.observation = msg
        self.get_logger().info(
            f'published u={msg.u} v={msg.v} stamp={frame.sec}.{frame.nanosec:09d}')
        return response

    def publish_status(self):
        result = self.monitor.report(self.now_ns())
        latest = self.registry.latest()
        status = DiagnosticStatus()
        status.level = LEVELS.get(result['state'], DiagnosticStatus.WARN)
        status.name = f'ffw_vision/{self.config.camera_id}/input'
        status.message = result['state']
        status.hardware_id = self.config.image_topic
        values = {
            'rate_hz': f"{result['rate_hz']:.1f}",
            'images': result['images'],
            'camera_infos': result['info'][0],
            'unpaired_camera_infos': result['info'][2],
            'latency_avg_ms': f"{result['latency_avg_ms']:.1f}",
            'last_image_stamp': f'{latest.sec}.{latest.nanosec:09d}' if latest else '',
            'errors': ','.join(f'{k}:{v}' for k, v in result['errors'].items()),
        }
        status.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [status]
        self.status_pub.publish(array)


def main(args=None):
    # Ctrl+C raises KeyboardInterrupt only; shutdown happens once below
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = PixelObserverNode()
    except ValueError as error:
        get_logger('pixel_observer').error(str(error))
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
