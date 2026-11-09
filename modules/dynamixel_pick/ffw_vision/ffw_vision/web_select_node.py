from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading

from ament_index_python.packages import get_package_share_directory
import cv2
from ffw_vision.camera_config import declare_camera_parameters, load_camera_config
from ffw_vision.depth_stats import color_from_msg
from ffw_vision.display_mapping import display_to_source
from ffw_vision.frame_registry import FrameInfo, FrameRegistry
from ffw_vision.input_check import NS_PER_SEC
from ffw_vision_msgs.srv import SelectPixel
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import String

SERVICE_TIMEOUT_SEC = 2.0
SPIN_TIMEOUT_SEC = 0.1


class WebSelectNode(Node):
    """Browser view of one camera; sends clicks on the shown frame to pixel_observer."""

    def __init__(self):
        super().__init__('web_select')
        declare_camera_parameters(self)
        self.declare_parameter('bind_address', '127.0.0.1')
        self.declare_parameter('port', 8765)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('retention_sec', 120.0)
        self.declare_parameter('display_topic', '')  # same stamps as image_topic, e.g. debug_image
        self.declare_parameter('info_topic', '')  # std_msgs/String shown next to the image
        self.config = load_camera_config(self)
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.registry = FrameRegistry(int(self.get_parameter('retention_sec').value * NS_PER_SEC))
        self.lock = threading.Lock()
        self.latest = None  # (FrameInfo, BGR image)
        page_path = os.path.join(get_package_share_directory('ffw_vision'), 'web', 'index.html')
        with open(page_path, 'rb') as page:
            self.page = page.read()

        self.client = self.create_client(SelectPixel, 'select_pixel')
        display_topic = self.get_parameter('display_topic').value or self.config.image_topic
        self.create_subscription(Image, display_topic, self.on_image, qos_profile_sensor_data)
        self.info_text = None
        if self.get_parameter('info_topic').value:
            self.create_subscription(String, self.get_parameter('info_topic').value,
                                     self.on_info, 10)
        address =(self.get_parameter('bind_address').value, self.get_parameter('port').value)
        self.server = ThreadingHTTPServer(address, make_handler(self))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.get_logger().info(f'serving http://{address[0]}:{address[1]} for {self.config}')

    def on_image(self, msg):
        info = FrameInfo(msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id,
                         msg.width, msg.height, self.get_clock().now().nanoseconds)
        image = color_from_msg(msg)
        with self.lock:
            self.registry.add(info)
            self.latest = (info, image)

    def on_info(self, msg):
        with self.lock:
            self.info_text = msg.data

    def latest_jpeg(self):
        with self.lock:
            latest = self.latest
        if latest is None:
            return None, None
        ok, data = cv2.imencode(
            '.jpg', latest[1], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return (latest[0], data.tobytes()) if ok else (None, None)

    def select(self, request):
        stamp_ns = int(request['stamp_sec']) * NS_PER_SEC + int(request['stamp_nanosec'])
        with self.lock:
            info = self.registry.get(stamp_ns)
        if info is None:
            raise LookupError('frame not found: expired or not from this camera')
        u, v = display_to_source(
            float(request['x']), float(request['y']),
            (float(request['display_width']), float(request['display_height'])),
            (info.width, info.height))
        response = self.call_select(info, u, v)
        obs = response.observation
        result = {
            'accepted': response.accepted,
            'message': response.message,
            'requested_u': u,
            'requested_v': v,
            'camera_id': obs.camera_id,
            'image_topic': obs.image_topic,
            'frame_id': obs.header.frame_id,
            'stamp_sec': obs.header.stamp.sec,
            'stamp_nanosec': obs.header.stamp.nanosec,
            'image_width': obs.image_width,
            'image_height': obs.image_height,
            'u': obs.u,
            'v': obs.v,
        }
        self.get_logger().info(f'selection {result}')
        return result

    def call_select(self, info, u, v):
        if not self.client.service_is_ready():
            raise ConnectionError(f'service {self.client.srv_name} is not available')
        request = SelectPixel.Request()
        request.camera_id = self.config.camera_id
        request.image_stamp.sec = info.sec
        request.image_stamp.nanosec = info.nanosec
        request.u = u
        request.v = v
        done = threading.Event()
        future = self.client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(SERVICE_TIMEOUT_SEC):
            raise ConnectionError(f'service {self.client.srv_name} did not answer')
        return future.result()

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()


def make_handler(node):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def send(self, code, body, content_type, headers=None):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, code, payload):
            self.send(code, json.dumps(payload).encode(), 'application/json')

        def do_GET(self):
            if self.path == '/':
                self.send(200, node.page, 'text/html; charset=utf-8')
            elif self.path.startswith('/frame'):
                info, jpeg = node.latest_jpeg()
                if info is None:
                    self.send_json(503, {'error': 'no frame yet'})
                    return
                self.send(200, jpeg, 'image/jpeg', {
                    'X-Stamp-Sec': info.sec, 'X-Stamp-Nanosec': info.nanosec,
                    'X-Frame-Id': info.frame_id, 'X-Width': info.width,
                    'X-Height': info.height, 'X-Camera-Id': node.config.camera_id})
            elif self.path.startswith('/info'):
                with node.lock:
                    text = node.info_text
                if text is None:
                    self.send_json(404, {'error': 'no info topic or no message yet'})
                else:
                    self.send(200, text.encode('utf-8'), 'text/plain; charset=utf-8')
            else:
                self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/select':
                self.send_json(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                self.send_json(200, node.select(json.loads(self.rfile.read(length))))
            except LookupError as error:
                self.send_json(409, {'error': str(error)})
            except ConnectionError as error:
                self.send_json(503, {'error': str(error)})
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(400, {'error': str(error)})

    return Handler


def main(args=None):
    # Ctrl+C raises KeyboardInterrupt only; shutdown happens once below
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = WebSelectNode()
    except (ValueError, OSError) as error:
        get_logger('web_select').error(str(error))
        rclpy.try_shutdown()
        return 1
    try:
        # wait with a timeout: without signal handlers Ctrl+C is seen only between waits
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=SPIN_TIMEOUT_SEC)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
