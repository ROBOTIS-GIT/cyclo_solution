import math
import os
import time

from ament_index_python.packages import get_package_share_directory
from ffw_vision.blob_detector import detect, DetectorParams
from ffw_vision.camera_config import declare_camera_parameters, load_camera_config
from ffw_vision.depth_stats import color_from_msg, depth_from_msg
from ffw_vision.hybrid_detector import detect_hybrid, HybridParams
from ffw_vision.input_check import stamp_to_ns
from ffw_vision.obb_detector import yolo_obb_detector
from ffw_vision.observation import make_object_array
from ffw_vision.overlay import ANGLE_SOURCE_TEXT, draw_blobs, draw_seg_objects
from ffw_vision.seg_detector import (
    detect_seg, pixel_rays, resolve_model, SegParams, yolo_segmenter)
from ffw_vision_msgs.msg import ObjectObservationArray
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

BUFFER_SIZE = 30
LOG_PERIOD_SEC = 2.0
SPIN_TIMEOUT_SEC = 0.1


def image_msg(header, bgr):
    msg = Image()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


def seg_summary(o):
    source = ANGLE_SOURCE_TEXT.get(o.angle_source)  # hybrid direction source
    return (f'({o.u:.0f},{o.v:.0f} {o.length_m * 1000:.0f}x{o.width_m * 1000:.0f}mm {o.pose_label}'
            f' h={o.plane_height * 1000:.0f} s={o.score:.2f}'
            f'{" rect" if o.angle_source == "min_area_rect" else ""}'
            f'{" dir=" + source if source else ""}'
            f'{" AMBIGUOUS" if o.ambiguous else ""})')


def info_rows(objects):
    """Info panel table: one row per object, numbered as on the debug image."""
    rows = [' #  pose      top mm  size mm  score  dir']
    for i, o in enumerate(objects, 1):
        rows.append(f'{i:2d}  {o.pose_label:8s}  {o.plane_height * 1000:6.0f}  '
                    f'{o.length_m * 1000:3.0f}x{o.width_m * 1000:<3.0f}  {o.score:5.2f}  '
                    f'{ANGLE_SOURCE_TEXT.get(o.angle_source, o.angle_source[:4]):3s}'
                    f'{"  AMBIGUOUS" if o.ambiguous else ""}')
    return rows


def declare_dataclass(node, defaults):
    """Declare one parameter per field and build the dataclass from their values."""
    for name, value in vars(defaults).items():
        node.declare_parameter(name, list(value) if isinstance(value, tuple) else value)
    values = {name: node.get_parameter(name).value for name in vars(defaults)}
    return type(defaults)(**{name: tuple(v) if isinstance(v, list) else v
                             for name, v in values.items()})


def blob_summary(b):
    # h = top surface median / top level in mm, f = local floor, shift = color blob center offset
    shift = math.hypot(b.u - b.color_box[0], b.v - b.color_box[1]) if b.color_box else 0.0
    return (f'({b.u:.0f},{b.v:.0f} {b.length:.0f}x{b.width:.0f} {b.pose_label}'
            f' h={b.height * 1000:.0f}/{b.top_height * 1000:.0f} f={b.floor:.3f}'
            f' shift={shift:.1f} a={b.area}{" AMBIGUOUS" if b.ambiguous else ""})')


class ObjectDetectorNode(Node):
    """Color + aligned depth in, ObjectObservationArray out; rule-based or YOLO-seg detector."""

    def __init__(self):
        super().__init__('object_detector')
        declare_camera_parameters(self)
        self.declare_parameter('target_label', 'object')
        self.declare_parameter('depth_scale', 0.001)  # meters per 16UC1 unit
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('detector', 'rule')  # rule, yolo_seg or yolo_hybrid
        self.config = load_camera_config(self)
        self.detector = self.get_parameter('detector').value
        if self.detector == 'rule':
            self.params = declare_dataclass(self, DetectorParams())
        elif self.detector in ('yolo_seg', 'yolo_hybrid'):
            hybrid = self.detector == 'yolo_hybrid'
            self.declare_parameter('model_path', '')  # launch override; empty = model_file
            self.declare_parameter('model_file', '')  # name in share/ffw_vision/models or abs path
            self.declare_parameter('imgsz', 448)  # same as training
            self.params = declare_dataclass(self, HybridParams() if hybrid else SegParams())
            if not self.config.depth_topic or not self.config.camera_info_topic:
                raise ValueError(f'{self.detector} needs depth_topic and camera_info_topic')
            models_dir = os.path.join(get_package_share_directory('ffw_vision'), 'models')
            self.segment = self.load_model(
                yolo_segmenter, self.get_parameter('model_path').value,
                self.get_parameter('model_file').value, models_dir,
                self.get_parameter('imgsz').value, 'seg')
            self.boxes = None
            p = self.params
            self.info_head = [
                f'detector  {self.detector}   seg {os.path.basename(self.model_name)}'
                f' (imgsz {self.get_parameter("imgsz").value})',
                f'settings  conf {p.conf:.2f}   edge_skip_blue'
                f' {"on" if p.edge_skip_blue else "off"}']
            if hybrid:  # the OBB top-face model gives position, the seg model the direction
                self.declare_parameter('obb_model_path', '')
                self.declare_parameter('obb_model_file', '')
                self.declare_parameter('obb_imgsz', 640)
                self.boxes = self.load_model(
                    yolo_obb_detector, self.get_parameter('obb_model_path').value,
                    self.get_parameter('obb_model_file').value, models_dir,
                    self.get_parameter('obb_imgsz').value, 'obb')
                self.info_head.insert(1, f'          obb {os.path.basename(self.model_name)}'
                                         f' (imgsz {self.get_parameter("obb_imgsz").value})')
                self.info_head[-1] += (f'   window {p.hybrid_window_deg:g} deg   match'
                                       f' {p.hybrid_match_px:g} px   merge {p.box_merge_px:g} px')
        else:
            raise ValueError(f'unknown detector: {self.detector}')
        self.camera_info = None
        self.rays = None  # (key, rays) cached per CameraInfo
        self.target_label = self.get_parameter('target_label').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.images = {}
        self.depths = {}

        self.array_pub = self.create_publisher(
            ObjectObservationArray, 'object_observations', 10)
        self.debug_pub = self.info_pub = None
        if self.get_parameter('publish_debug_image').value:
            self.debug_pub = self.create_publisher(Image, 'debug_image', qos_profile_sensor_data)
            # settings and per-object numbers for the browser page, next to the image
            self.info_pub = self.create_publisher(String, 'debug_info', 10)
        self.create_subscription(
            Image, self.config.image_topic, self.on_image, qos_profile_sensor_data)
        if self.config.depth_topic:
            self.create_subscription(
                Image, self.config.depth_topic, self.on_depth, qos_profile_sensor_data)
        if self.detector in ('yolo_seg', 'yolo_hybrid'):
            self.create_subscription(
                CameraInfo, self.config.camera_info_topic, self.on_camera_info,
                qos_profile_sensor_data)
        self.get_logger().info(
            f'detecting {self.target_label} with {self.params} on {self.config}')

    def load_model(self, loader, path, file, models_dir, imgsz, kind):
        model = resolve_model(path, file, models_dir)
        if not os.path.isfile(model):
            raise ValueError(f'YOLO {kind} weights not found: {model}')
        self.get_logger().info(f'YOLO {kind} weights {model}, imgsz {imgsz}')
        self.model_name = model
        return loader(model, imgsz)

    def on_camera_info(self, msg):
        self.camera_info = msg

    def rays_for(self, info):
        key = (info.width, info.height, tuple(info.k), tuple(info.d))
        if self.rays is None or self.rays[0] != key:
            self.rays = (key, pixel_rays(info.width, info.height, info.k, info.d))
        return self.rays[1]

    def on_image(self, msg):
        if not self.config.depth_topic:
            self.process(msg, None)
            return
        self.store(self.images, msg)

    def on_depth(self, msg):
        self.store(self.depths, msg)

    def store(self, buffer, msg):
        stamp = stamp_to_ns(msg.header.stamp)
        buffer[stamp] = msg
        if len(buffer) > BUFFER_SIZE:
            del buffer[min(buffer)]
        if stamp in self.images and stamp in self.depths:
            self.process(self.images.pop(stamp), self.depths.pop(stamp))

    def process(self, color_msg, depth_msg):
        if self.detector in ('yolo_seg', 'yolo_hybrid'):
            self.process_seg(color_msg, depth_msg)
            return
        try:
            color = color_from_msg(color_msg)
            depth = depth_from_msg(depth_msg, self.depth_scale) if depth_msg else None
            blobs, floor = detect(color, depth, self.params)
        except ValueError as error:
            self.get_logger().error(str(error), throttle_duration_sec=LOG_PERIOD_SEC)
            return
        self.array_pub.publish(make_object_array(
            self.config, color_msg.header, color_msg.width, color_msg.height, blobs,
            self.target_label))
        if self.debug_pub is not None:
            self.debug_pub.publish(image_msg(
                color_msg.header, draw_blobs(color, blobs, self.params.roi, floor)))
        summary = ' '.join(blob_summary(b) for b in blobs)
        self.get_logger().info(
            f'floor={floor:.3f}m n={len(blobs)} {summary}', throttle_duration_sec=LOG_PERIOD_SEC)

    def process_seg(self, color_msg, depth_msg):
        info = self.camera_info
        if info is None:
            self.get_logger().warn('waiting for camera_info', throttle_duration_sec=LOG_PERIOD_SEC)
            return
        try:
            color = color_from_msg(color_msg)
            depth = depth_from_msg(depth_msg, self.depth_scale)
            if (info.width, info.height) != (color_msg.width, color_msg.height):
                raise ValueError(f'camera_info {info.width}x{info.height} does not match image')
            start = time.perf_counter()
            segments = self.segment(color, self.params.conf)
            boxes = None if self.boxes is None else self.boxes(color, self.params.conf)
            model_ms = (time.perf_counter() - start) * 1000
            if self.boxes is None:
                objects, plane = detect_seg(color, depth, info.k, info.d, segments, self.params,
                                            self.rays_for(info))
            else:
                objects, plane = detect_hybrid(
                    color, depth, info.k, info.d, segments, boxes, self.params,
                    self.rays_for(info))
            total_ms = (time.perf_counter() - start) * 1000
        except ValueError as error:
            self.get_logger().error(str(error), throttle_duration_sec=LOG_PERIOD_SEC)
            return
        self.array_pub.publish(make_object_array(
            self.config, color_msg.header, color_msg.width, color_msg.height, objects,
            self.target_label, plane))
        if self.debug_pub is not None:
            self.debug_pub.publish(image_msg(
                color_msg.header, draw_seg_objects(color, objects, plane is not None)))
            plane_text = f'camera above plane {plane[1]:.3f} m' if plane else 'no plane/basket'
            self.info_pub.publish(String(data='\n'.join(
                self.info_head + [f'frame     {plane_text}   objects {len(objects)}   models'
                                  f' {model_ms:.0f} ms   total {total_ms:.0f} ms', '']
                + info_rows(objects))))
        summary = ' '.join(seg_summary(o) for o in objects)
        plane_text = f'{plane[1]:.3f}m' if plane else '-'
        self.get_logger().info(
            f'camera above plane={plane_text} n={len(objects)} {summary}',
            throttle_duration_sec=LOG_PERIOD_SEC)


def main(args=None):
    # Ctrl+C raises KeyboardInterrupt only; shutdown happens once below
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = ObjectDetectorNode()
    except (ValueError, TypeError) as error:
        get_logger('object_detector').error(str(error))
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
