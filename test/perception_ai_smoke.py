"""Replay a measured head RGB/depth pair in an isolated container; no robot topics."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from demo_1f.msg import CupObservation
from rclpy.action import get_action_names_and_types
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

assert os.environ.get("DEMO_ISOLATED_TEST") == "1"
root = Path(sys.argv[1])
backend = sys.argv[2] if len(sys.argv) > 2 else "opencv_cpu"
rgb = cv2.imread(str(root / "head.png"))
depth = np.load(root / "depth.npy")
calibration = json.loads((root / "info.json").read_text())
rclpy.init()
node = rclpy.create_node("ai_replay_test")
bridge = CvBridge()
pubs = {
    key: node.create_publisher(cls, f"/ai_test/{key}", qos_profile_sensor_data)
    for key, cls in [("rgb", Image), ("depth", Image), ("info", CameraInfo)]
}
observations, images = [], []
node.create_subscription(CupObservation, "/perception_test_head/cup", observations.append, 10)
node.create_subscription(
    Image, "/perception_test_head/head/image", images.append, qos_profile_sensor_data
)
process = subprocess.Popen(
    [
        "ros2",
        "launch",
        "demo_1f",
        "perception_test.launch.py",
        "rgb_topic:=/ai_test/rgb",
        "depth_topic:=/ai_test/depth",
        "info_topic:=/ai_test/info",
        f"ai_backend:={backend}",
    ],
    start_new_session=True,
)


def run_phase(phase, predicate, seconds=25):
    observations.clear()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("Perception exited")
        image = bridge.cv2_to_imgmsg(rgb if phase != "blank" else np.full_like(rgb, 230), "bgr8")
        image.header.stamp = node.get_clock().now().to_msg()
        image.header.frame_id = "head_optical"
        dep = depth.copy()
        if phase == "no_depth":
            dep[:] = np.nan if dep.dtype.kind == "f" else 0
        dm = bridge.cv2_to_imgmsg(dep, calibration["depth_encoding"])
        dm.header = image.header
        info = CameraInfo()
        info.header = image.header
        info.width, info.height = calibration["width"], calibration["height"]
        info.p = calibration["p"]
        for key, msg in [("info", info), ("depth", dm), ("rgb", image)]:
            pubs[key].publish(msg)
        rclpy.spin_once(node, timeout_sec=0.04)
        matching = [m for m in observations if predicate(m)]
        if len(matching) >= 5:
            return matching[-1]
    raise AssertionError([(m.valid, m.detail) for m in observations[-10:]])


try:
    result = run_phase("real", lambda m: m.valid and m.observation_type == CupObservation.BODY)
    assert result.source == "head" and result.confidence >= 0.15
    assert "AI_BODY_COARSE" in result.detail
    assert not get_action_names_and_types(node)
    assert "/perception/cup" not in dict(node.get_topic_names_and_types())
    if images:
        cv2.imwrite(str(root / f"ai-{backend}.png"), bridge.imgmsg_to_cv2(images[-1], "bgr8"))
    print("REAL_FRAME", result, flush=True)
    run_phase("no_depth", lambda m: not m.valid and "insufficient mask depth" in m.detail)
    run_phase("blank", lambda m: not m.valid and "no cup" in m.detail)
    print(
        f"PASS {backend}: real RGB/depth BODY, missing depth rejected, blank rejected; no motion",
        flush=True,
    )
finally:
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    node.destroy_node()
    rclpy.shutdown()
