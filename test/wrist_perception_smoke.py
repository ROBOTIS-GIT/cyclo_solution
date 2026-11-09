"""Exercise wrist perception publication contracts without a camera/robot."""

import os
import sys
from unittest.mock import patch
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo

assert os.environ.get("DEMO_ISOLATED_TEST") == "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from action_runner import selection  # noqa: E402
from wrist_perception import WristPerception  # noqa: E402

rclpy.init()
node = WristPerception()
results = []
node.publish = lambda message, detail, frame=None, fit=None: results.append((detail, fit))
try:
    info = CameraInfo()
    info.width, info.height = 320, 240
    info.p = [200.0, 0.0, 160.0, 0.0, 0.0, 200.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    bridge = CvBridge()
    empty = bridge.cv2_to_imgmsg(np.zeros((240, 320), np.uint16), "16UC1")
    node.detect(empty, info, transform)
    assert results[-1][0].startswith("NO_CANDIDATE:") and results[-1][1] is None
    # A flat disk is a region, not evidence of a rim; do not publish its centroid.
    import cv2

    depth = np.zeros((240, 320), np.uint16)
    cv2.circle(depth, (160, 120), 32, 300, -1)
    node.detect(bridge.cv2_to_imgmsg(depth, "16UC1"), info, transform)
    assert results[-1][0].startswith("RIM_UNCERTAIN:") and results[-1][1] is None
    # A model-based center is still withheld until three consistent frames arrive.
    candidate = {"contours": [], "pixel": [160, 120]}
    fit = {"center": np.array([0., 0., .3]), "radius": .0475,
           "points": np.array([[0., 0., .3]]), "coverage_deg": 180.,
           "error": .012, "method": "known_radius"}
    def detection(*args):
        current = {**fit, "center": fit["center"].copy()}
        return candidate, current, "RIM_FIT_KNOWN_RADIUS", [(candidate, current, "")]
    with patch("wrist_perception.cup_candidates", return_value=[candidate]), \
            patch("wrist_perception.fit_rim_candidates", side_effect=detection):
        for i in range(3):
            message = bridge.cv2_to_imgmsg(depth, "16UC1")
            message.header.stamp = node.get_clock().now().to_msg()
            node.detect(message, info, transform)
            assert (results[-1][1] is not None) == (i == 2)
        assert "method=known_radius stable=3/3" in results[-1][0]
        fit["center"][0] += .02
        message.header.stamp = node.get_clock().now().to_msg()
        node.detect(message, info, transform)
        assert results[-1][0] == "RIM_UNSTABLE: 1/3" and results[-1][1] is None
    assert selection("1", "cup")[1].command == "initial"
    assert selection("2", "cup")[1].command == "pick"
    assert selection("3", "cup")[1].direction == "right_to_left"
    assert selection("4", "cup")[0] == "approach" and selection("4", "cup")[1].arm == "left"
    assert selection("5", "cup")[0] == "refine" and selection("5", "cup")[1].execute
    assert selection("6", "cup")[0] == "withdraw" and selection("6", "cup")[1].arm == "both"
    assert selection("d", "cup")[1].command == "start"
    assert selection("r", "cup")[1].command == "resume"
    for removed in ("0", "7", "8", "9", "10", "11", "i", "k", "g", "s", "h"):
        assert selection(removed, "cup") is None
    print("PASS: missing cup and unconfirmed rim never produce a valid target; runner modes")
finally:
    node.destroy_node()
    rclpy.shutdown()
