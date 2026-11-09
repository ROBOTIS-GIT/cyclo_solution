#!/usr/bin/env python3
"""Wrist rim depth perception. Publishes observations, never commands."""

from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from demo_1f.msg import CupObservation
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from wrist_cup_geometry import cup_candidates, fit_rim_candidates, rotation_matrix


class WristPerception(Node):
    def __init__(self):
        super().__init__("wrist_cup")
        self.arm = self.declare_parameter("arm", "left").value
        if self.arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        self.filter_raw = self.declare_parameter("filter_raw", self.arm == "right").value
        self.near = self.declare_parameter("depth_min_m", 0.18).value
        self.far = self.declare_parameter("depth_max_m", 0.38).value
        self.exclude = self.declare_parameter("exclude_rect", [0.0, 0.32, 0.175, 0.69]).value
        if not (
            0 < self.near < self.far
            and len(self.exclude) == 4
            and all(np.isfinite(v) and 0 <= v <= 1 for v in self.exclude)
            and self.exclude[0] <= self.exclude[2]
            and self.exclude[1] <= self.exclude[3]
        ):
            raise ValueError("Invalid numeric depth filter or exclusion rectangle")
        self.frame = self.declare_parameter("planning_frame", "base_link").value
        self.inner = self.declare_parameter("inner_radius", 0.045).value
        self.wall = self.declare_parameter("wall_thickness", 0.005).value
        if not (
            np.isfinite(self.inner)
            and 0.02 <= self.inner <= 0.15
            and np.isfinite(self.wall)
            and 0 < self.wall <= 0.02
        ):
            raise ValueError("Invalid wrist cup dimensions")
        self.cup_id = self.declare_parameter("cup_id", "cup").value
        self.bridge = CvBridge()
        self.info = None
        self.pending = deque(maxlen=8)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.history = deque(maxlen=3)
        self.last_stamp = 0
        self.observations = self.create_publisher(CupObservation, "/perception/wrist_cup", 5)
        self.images = self.create_publisher(Image, "/wrist_cup/image", qos_profile_sensor_data)
        self.status = self.create_publisher(String, "/wrist_cup/status", 1)
        topic = self.declare_parameter(
            "depth_topic",
            (
                "/perception_viewer_head/left/depth_filtered"
                if self.arm == "left"
                else f"/camera_{self.arm}/camera_{self.arm}/depth/image_rect_raw"
            ),
        ).value
        self.create_subscription(
            Image, topic, lambda message: self.pending.append(message),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        info_topic = self.declare_parameter(
            "info_topic", f"/camera_{self.arm}/camera_{self.arm}/depth/camera_info"
        ).value
        self.create_subscription(CameraInfo, info_topic, self.camera_info, qos_profile_sensor_data)
        self.create_timer(0.02, self.process)

    def camera_info(self, message):
        self.info = message

    def publish(self, message, detail, frame=None, fit=None):
        # A slow fit must not turn an expired image into a fresh motion target.
        if fit is not None:
            stamp = rclpy.time.Time.from_msg(message.header.stamp)
            age = (self.get_clock().now() - stamp).nanoseconds / 1e9
            if age < -0.1 or age > 0.8:
                self.history.clear()
                fit = None
                detail = f"STALE_FRAME: rim result image age={age:.3f}s"
        observation = CupObservation()
        observation.header = message.header
        observation.header.frame_id = self.frame
        observation.source = "wrist_" + self.arm
        observation.cup_id = self.cup_id
        observation.observation_type = 2
        observation.inner_radius = self.inner
        observation.detail = detail
        if fit is not None:
            observation.valid = True
            (
                observation.body_position.x,
                observation.body_position.y,
                observation.body_position.z,
            ) = [float(v) for v in fit["center"]]
            observation.position_error = fit["error"]
            observation.confidence = min(1.0, fit["coverage_deg"] / 360.0)
        self.observations.publish(observation)
        self.status.publish(String(data=detail))
        if frame is not None:
            image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            image.header = message.header
            self.images.publish(image)

    def process(self):
        # Fit at most ONE image, then yield to image/TF callbacks. Draining the queue
        # here starves TF and lets several expensive fits age every pending image.
        queued = sorted(self.pending,
                        key=lambda m: (m.header.stamp.sec, m.header.stamp.nanosec),
                        reverse=True)
        self.pending.clear()
        waiting, diagnostic = [], None
        for message in queued:
            stamp = rclpy.time.Time.from_msg(message.header.stamp)
            age = (self.get_clock().now() - stamp).nanoseconds / 1e9
            if stamp.nanoseconds <= self.last_stamp or age < -0.1 or age > 0.8:
                continue
            info = self.info
            if (
                info is None
                or info.header.frame_id != message.header.frame_id
                or (info.height, info.width) != (message.height, message.width)
                or not all(np.isfinite(info.p))
                or info.p[0] <= 0
                or info.p[5] <= 0
            ):
                if diagnostic is None:
                    diagnostic = (message, "CAMERA_INFO_INVALID")
                continue
            try:
                transform = self.buffer.lookup_transform(self.frame, message.header.frame_id, stamp)
            except TransformException:
                if age <= 0.65:
                    waiting.append(message)
                elif diagnostic is None:
                    diagnostic = (message, "TF_UNAVAILABLE")
                continue
            # Keep newer frames still waiting for THEIR timestamp's TF, discard older
            # backlog. A future TF for the newest image must not block an older ready one.
            self.pending.extend(reversed(waiting))
            self.last_stamp = stamp.nanoseconds
            try:
                self.detect(message, info, transform)
            except (ValueError, cv2.error) as error:
                self.history.clear()
                self.publish(message, f"DEPTH_INVALID: {error}")
            return
        self.pending.extend(reversed(waiting))
        if diagnostic is not None:
            self.history.clear()
            self.publish(*diagnostic)

    def detect(self, message, info, transform):
        if message.encoding.lower() not in ("16uc1", "32fc1"):
            raise ValueError("Numeric 16UC1/32FC1 depth required")
        meters = self.bridge.imgmsg_to_cv2(message).astype(np.float32)
        if message.encoding.lower() == "16uc1":
            meters *= 0.001
        keep = np.isfinite(meters) & (meters > 0)
        if self.filter_raw:
            keep &= (meters >= self.near) & (meters <= self.far)
            h, w = meters.shape
            x0, y0, x1, y1 = self.exclude
            keep[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = False
            meters[~keep] = 0
        display = np.nan_to_num(np.clip((0.5 - meters) / 0.4, 0, 1), nan=0) * 255
        display[~keep] = 0
        frame = cv2.cvtColor(display.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        intrinsics = (info.p[0], info.p[5], info.p[2], info.p[6])
        candidates = cup_candidates(meters, keep, intrinsics, 2 * (self.inner + self.wall))
        if not candidates:
            self.history.clear()
            self.publish(message, "NO_CANDIDATE: no cup-sized region after gripper mask", frame)
            return
        q, t = transform.transform.rotation, transform.transform.translation
        rotation = rotation_matrix([q.x, q.y, q.z, q.w])
        translation = np.array([t.x, t.y, t.z])
        candidate, fit, detail, evaluated = fit_rim_candidates(
            meters, candidates, intrinsics, rotation, translation, self.inner, self.wall
        )
        for index, (region, region_fit, _) in enumerate(evaluated):
            color = (0, 220, 0) if region_fit is not None else (0, 80, 220)
            cv2.drawContours(frame, region["contours"], -1, color, 1)
            cv2.putText(frame, str(index + 1), tuple(region["pixel"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if fit is None:
            self.history.clear()
            self.publish(message, detail, frame)
            return

        def pixels(world_points):
            camera = (world_points - translation) @ rotation
            camera = camera[camera[:, 2] > 0.03]
            return np.column_stack(
                (
                    camera[:, 0] / camera[:, 2] * info.p[0] + info.p[2],
                    camera[:, 1] / camera[:, 2] * info.p[5] + info.p[6],
                )
            ).astype(int)

        for point in pixels(fit["points"])[::3]:
            cv2.circle(frame, tuple(point), 1, (255, 120, 0), -1)
        angles = np.linspace(0, 2 * np.pi, 100)
        ring = np.tile(fit["center"], (100, 1))
        ring[:, 0] += fit["radius"] * np.cos(angles)
        ring[:, 1] += fit["radius"] * np.sin(angles)
        projected = pixels(ring)
        if len(projected) == 100:
            cv2.polylines(frame, [projected], True, (255, 0, 255), 1)
        center_pixel = pixels(fit["center"][None, :])
        if len(center_pixel):
            cv2.drawMarker(frame, tuple(center_pixel[0]), (255, 0, 255), cv2.MARKER_CROSS, 15, 2)
        stamp = rclpy.time.Time.from_msg(message.header.stamp).nanoseconds
        if self.history and (
            stamp - self.history[-1][0] > 600_000_000
            or max(np.linalg.norm(fit["center"] - p) for _, p in self.history) > 0.008
        ):
            self.history.clear()
        self.history.append((stamp, fit["center"].copy()))
        if len(self.history) < 3:
            self.publish(message, f"RIM_UNSTABLE: {len(self.history)}/3", frame)
            return
        fit["center"] = np.mean([p for _, p in self.history], axis=0)
        detail = (
            f"VALID_RIM: XYZ={np.round(fit['center'], 4).tolist()} "
            f"radius={fit['radius']:.3f}m arc={fit['coverage_deg']:.0f}deg "
            f"method={fit['method']} stable=3/3; no motion command"
        )
        self.publish(message, detail, frame, fit)


def main():
    rclpy.init()
    node = WristPerception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
