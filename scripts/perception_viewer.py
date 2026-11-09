#!/usr/bin/env python3
"""Read-only, dependency-light browser viewer for the isolated perception test."""

import math
import signal
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from demo_perception.depth import depth_meters, filter_depth, gray_depth
from demo_perception.viewer_http import make_handler
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from wrist_cup_geometry import cup_candidates


class Viewer(Node):
    def __init__(self):
        super().__init__("perception_viewer")
        self.source = "head"
        prefix = self.declare_parameter("topic_prefix", "/perception_test_head").value
        self.host = self.declare_parameter("host", "127.0.0.1").value
        self.port = self.declare_parameter("port", 8765).value
        self.lock = threading.Lock()
        self.jpeg = None
        self.status = "WAITING: perception_test node"
        self.status_received = None
        self.image_received = None
        self.image_stamp = None
        self.bridge = CvBridge()
        self.depth = {}
        self.filter_near = self.declare_parameter("left_depth_min_m", 0.22).value
        self.filter_far = self.declare_parameter("left_depth_max_m", 0.38).value
        self.check_filter(self.filter_near, self.filter_far)
        self.filtered = {"jpeg": None, "received": None, "detail": "숫자 depth 수신 대기"}
        self.left_info = None
        self.candidate_previous = None
        self.candidate_count = 0
        self.candidate = {
            "jpeg": None,
            "received": None,
            "detail": "depth / CameraInfo 수신 대기",
            "selected": None,
            "count": 0,
            "stable_frames": 0,
            "frame_id": "",
        }
        self.cup_diameter = self.declare_parameter("candidate_diameter_m", 0.10).value
        if not math.isfinite(self.cup_diameter) or self.cup_diameter <= 0:
            raise ValueError("candidate_diameter_m must be positive")
        info_topic = self.declare_parameter(
            "depth_left_info_topic", "/camera_left/camera_left/depth/camera_info"
        ).value
        self.create_subscription(CameraInfo, info_topic, self.camera_info, qos_profile_sensor_data)
        self.exclude_rect = self.declare_parameter(
            "left_exclude_rect", [0.0, 0.32, 0.23, 0.72]
        ).value
        if (
            len(self.exclude_rect) != 4
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in self.exclude_rect)
            or self.exclude_rect[0] >= self.exclude_rect[2]
            or self.exclude_rect[1] >= self.exclude_rect[3]
        ):
            raise ValueError("left_exclude_rect must be normalized [x0,y0,x1,y1]")
        self.rim = {"jpeg": None, "received": None, "detail": "손목 입구 추정 노드 대기"}
        self.create_subscription(Image, "/wrist_cup/image", self.rim_image, qos_profile_sensor_data)
        self.create_subscription(String, "/wrist_cup/status", self.rim_status, 1)
        self.filtered_publisher = self.create_publisher(
            Image, "~/left/depth_filtered", qos_profile_sensor_data
        )
        for name, default in {
            "head": "/zed/zed_node/depth/depth_registered",
            "left": "/camera_left/camera_left/depth/image_rect_raw",
            "right": "/camera_right/camera_right/depth/image_rect_raw",
        }.items():
            topic = self.declare_parameter(f"depth_{name}_topic", default).value
            self.depth[name] = {"topic": topic, "jpeg": None, "received": None, "encoding": None}
            self.create_subscription(
                Image,
                topic,
                lambda message, camera=name: self.depth_image(camera, message),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            Image, f"{prefix}/{self.source}/image", self.image, qos_profile_sensor_data
        )
        self.create_subscription(
            String,
            f"{prefix}/{self.source}/status",
            self.message,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

    def rim_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, "bgr8")
        ok, jpeg = cv2.imencode(".jpg", frame)
        if ok:
            with self.lock:
                self.rim.update(jpeg=jpeg.tobytes(), received=time.monotonic())

    def rim_status(self, message):
        with self.lock:
            self.rim["detail"] = message.data

    def camera_info(self, message):
        with self.lock:
            self.left_info = message

    def candidate_frame(self, meters, keep, gray, header, near, far):
        with self.lock:
            info = self.left_info
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        height, width = meters.shape
        x0, y0, x1, y1 = self.exclude_rect
        cv2.rectangle(
            frame,
            (int(x0 * width), int(y0 * height)),
            (int(x1 * width), int(y1 * height)),
            (0, 0, 180),
            1,
        )
        candidates, selected = [], None
        detail = "정렬된 depth CameraInfo 대기"
        usable = (
            info is not None
            and info.header.frame_id == header.frame_id
            and (info.height, info.width) == meters.shape
            and all(math.isfinite(v) for v in info.p)
            and info.p[0] > 0
            and info.p[5] > 0
        )
        if usable:
            candidates = cup_candidates(
                meters, keep, (info.p[0], info.p[5], info.p[2], info.p[6]), self.cup_diameter
            )
            # Similar candidates remain ambiguous rather than selecting the largest blob.
            if candidates and (
                len(candidates) == 1 or candidates[0]["score"] > candidates[1]["score"] * 1.2
            ):
                selected = candidates[0]
            detail = (
                "크기 조건에 맞는 컵 후보 없음"
                if not candidates
                else "여러 크기 후보 · 아래 입구 추정에서 림 형상으로 판별"
            )
        with self.lock:
            if (near, far) != (self.filter_near, self.filter_far):
                return
            now = time.monotonic()
            stamp = (header.stamp.sec, header.stamp.nanosec)
            if selected is not None:
                previous = self.candidate_previous
                same = (
                    previous is not None
                    and now - previous["time"] < 0.8
                    and previous["frame"] == header.frame_id
                    and np.linalg.norm(np.array(selected["xyz"]) - previous["xyz"]) < 0.02
                )
                if not same:
                    self.candidate_count = 1
                elif stamp != previous["stamp"]:
                    self.candidate_count += 1
                self.candidate_previous = {
                    "xyz": np.array(selected["xyz"]),
                    "time": now,
                    "frame": header.frame_id,
                    "stamp": stamp,
                }
                x, y, z = selected["xyz"]
                detail = (
                    f"{'안정 후보' if self.candidate_count >= 3 else '후보 확인 중'} · "
                    f"{self.candidate_count}프레임 · 표면 XYZ=({x:.3f}, {y:.3f}, {z:.3f})m · {header.frame_id}"
                )
            else:
                self.candidate_count = 0
                self.candidate_previous = None
            for candidate in candidates:
                color = (
                    (0, 220, 0)
                    if candidate is selected and self.candidate_count >= 3
                    else (0, 220, 255)
                )
                cv2.drawContours(frame, candidate["contours"], -1, color, 2)
                if candidate is selected:
                    cv2.drawMarker(frame, tuple(candidate["pixel"]), color, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                frame,
                f"CANDIDATES: {len(candidates)}  STABLE: {self.candidate_count}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                public = (
                    None
                    if selected is None
                    else {k: v for k, v in selected.items() if k not in ("contours", "_mask")}
                )
                self.candidate.update(
                    jpeg=jpeg.tobytes(),
                    received=now,
                    detail=detail,
                    selected=public,
                    count=len(candidates),
                    stable_frames=self.candidate_count,
                    frame_id=header.frame_id,
                )

    def image(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, "bgr8")
            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, round(frame.shape[0] * 1280 / frame.shape[1])))
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.jpeg = jpeg.tobytes()
                    self.image_received = time.monotonic()
                    self.image_stamp = rclpy.time.Time.from_msg(message.header.stamp)
        except Exception as error:
            self.get_logger().warning(f"Image decode failed: {error}")

    @staticmethod
    def check_filter(near, far):
        if not all(math.isfinite(v) for v in (near, far)) or not 0 < near < far <= 5:
            raise ValueError("0 < 최소 < 최대 ≤ 500cm 범위로 입력하세요")

    def set_filter(self, near, far):
        self.check_filter(near, far)
        with self.lock:
            self.filter_near, self.filter_far = near, far
            self.filtered.update(jpeg=None, received=None, detail="새 거리 범위 적용 중")
            self.clear_candidate("새 거리 범위 적용 중")

    def clear_candidate(self, detail):
        # Caller holds self.lock.
        self.candidate_previous = None
        self.candidate_count = 0
        self.candidate.update(
            jpeg=None, received=None, selected=None, count=0, stable_frames=0, detail=detail
        )

    def depth_image(self, camera, message):
        with self.lock:
            last = self.depth[camera]["received"]
            near, far = self.filter_near, self.filter_far
        if last is not None and time.monotonic() - last < 0.2:
            return
        try:
            encoding = message.encoding.lower()
            if encoding in ("16uc1", "mono16", "32fc1"):
                raw = self.bridge.imgmsg_to_cv2(message, "passthrough")
                meters = depth_meters(raw, encoding)
                valid = np.isfinite(meters) & (meters > 0)
                frame = gray_depth(meters, valid, 0.1, 2.0)
                if camera == "left":
                    filtered, meters, keep = filter_depth(raw, encoding, near, far)
                    height, width = meters.shape
                    x0, y0, x1, y1 = self.exclude_rect
                    excluded = (
                        slice(int(y0 * height), int(y1 * height)),
                        slice(int(x0 * width), int(x1 * width)),
                    )
                    keep[excluded] = False
                    filtered[excluded] = 0
                    filtered_frame = gray_depth(meters, keep, near, far)
                    ok, jpeg = cv2.imencode(".jpg", filtered_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.candidate_frame(meters, keep, filtered_frame, message.header, near, far)
                    numeric = self.bridge.cv2_to_imgmsg(filtered, encoding=message.encoding)
                    numeric.header = message.header
                    with self.lock:
                        # A concurrent UI change must not label an old-range frame as new.
                        if (near, far) == (self.filter_near, self.filter_far) and ok:
                            self.filtered.update(
                                jpeg=jpeg.tobytes(),
                                received=time.monotonic(),
                                detail=f"남은 점 {np.count_nonzero(keep)} / {keep.size} · {message.encoding}",
                            )
                            self.filtered_publisher.publish(numeric)
            elif encoding in ("rgb8", "bgr8"):
                frame = self.bridge.imgmsg_to_cv2(message, "bgr8")
                if camera == "left":
                    with self.lock:
                        self.clear_candidate("원본 숫자 depth 필요")
                        self.filtered.update(
                            jpeg=None,
                            received=None,
                            detail="원본 depth 필요: 카메라 colorizer를 꺼주세요",
                        )
            else:
                raise ValueError(f"Unsupported depth encoding {message.encoding}")
            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, round(frame.shape[0] * 1280 / frame.shape[1])))
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.depth[camera].update(
                        jpeg=jpeg.tobytes(), received=time.monotonic(), encoding=message.encoding
                    )
        except Exception as error:
            self.get_logger().warning(
                f"{camera} depth decode failed: {error}", throttle_duration_sec=5.0
            )

    def message(self, message):
        with self.lock:
            self.status = message.data
            self.status_received = time.monotonic()

    def state(self):
        with self.lock:
            now = time.monotonic()
            return {
                "status": self.status,
                "camera": self.source,
                "rim": {
                    "has_image": self.rim["jpeg"] is not None,
                    "detail": self.rim["detail"],
                    "age": None if self.rim["received"] is None else now - self.rim["received"],
                },
                "candidate": {
                    **{k: v for k, v in self.candidate.items() if k not in ("jpeg", "received")},
                    "has_image": self.candidate["jpeg"] is not None,
                    "age": None
                    if self.candidate["received"] is None
                    else now - self.candidate["received"],
                },
                "filter": {
                    "min_m": self.filter_near,
                    "max_m": self.filter_far,
                    "has_image": self.filtered["jpeg"] is not None,
                    "detail": self.filtered["detail"],
                    "age": None
                    if self.filtered["received"] is None
                    else now - self.filtered["received"],
                },
                "depth": {
                    name: {
                        "topic": item["topic"],
                        "has_image": item["jpeg"] is not None,
                        "encoding": item["encoding"],
                        "age": None if item["received"] is None else now - item["received"],
                    }
                    for name, item in self.depth.items()
                },
                "has_image": self.jpeg is not None,
                "status_age": None if self.status_received is None else now - self.status_received,
                "image_age": None
                if self.image_stamp is None
                else (self.get_clock().now() - self.image_stamp).nanoseconds / 1e9,
            }


def main():
    rclpy.init()
    viewer = Viewer()
    server = None
    try:
        server = ThreadingHTTPServer((viewer.host, viewer.port), make_handler(viewer))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        viewer.get_logger().info(f"Read-only viewer: http://{viewer.host}:{viewer.port}")
        rclpy.spin(viewer)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        # A terminal interrupt may reach both the launch parent and this child.
        # Ignore a second forwarded SIGINT while closing the HTTP thread.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if server:
            server.shutdown()
            server.server_close()
        viewer.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
