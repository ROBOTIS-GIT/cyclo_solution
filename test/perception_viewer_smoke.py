"""Exercise the browser HTTP endpoints without cameras or robot connections."""

import importlib.util
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

assert os.environ.get("DEMO_ISOLATED_TEST") == "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "viewer", Path(__file__).resolve().parents[1] / "scripts/perception_viewer.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rclpy.init()
viewer = module.Viewer()
server = ThreadingHTTPServer(("127.0.0.1", 0), module.make_handler(viewer))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f"http://127.0.0.1:{server.server_port}"
try:
    with urlopen(base + "/", timeout=3) as response:
        assert response.status == 200
    with urlopen(base + "/status", timeout=3) as response:
        status = json.load(response)
    assert "body" not in status
    assert status["camera"] == "head" and not status["has_image"]
    assert status["status_age"] is None and status["image_age"] is None
    assert all(not state["has_image"] for state in status["depth"].values())
    try:
        urlopen(base + "/image.jpg", timeout=3)
        raise AssertionError("Missing image should return 503")
    except HTTPError as error:
        assert error.code == 503
    try:
        urlopen(base + "/depth/left.jpg", timeout=3)
        raise AssertionError("Missing depth should return 503")
    except HTTPError as error:
        assert error.code == 503
    bridge = CvBridge()
    viewer.depth_image(
        "head",
        bridge.cv2_to_imgmsg(
            np.array([[0.0, 0.5], [float("nan"), 1.0]], dtype=np.float32), "32FC1"
        ),
    )
    viewer.depth_image(
        "left", bridge.cv2_to_imgmsg(np.full((32, 32, 3), (255, 0, 0), dtype=np.uint8), "rgb8")
    )
    viewer.depth_image(
        "right", bridge.cv2_to_imgmsg(np.full((32, 32), 600, dtype=np.uint16), "16UC1")
    )
    with urlopen(base + "/status", timeout=3) as response:
        status = json.load(response)
    assert [status["depth"][name]["encoding"] for name in ("head", "left", "right")] == [
        "32FC1",
        "rgb8",
        "16UC1",
    ]
    for name in ("head", "left", "right"):
        with urlopen(base + f"/depth/{name}.jpg", timeout=3) as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read().startswith(b"\xff\xd8")
    frame = CvBridge().cv2_to_imgmsg(np.zeros((32, 32, 3), dtype=np.uint8), "bgr8")
    frame.header.stamp = viewer.get_clock().now().to_msg()
    viewer.image(frame)
    viewer.message(String(data="head: VALID: AI_BODY_COARSE"))
    with urlopen(base + "/status", timeout=3) as response:
        status = json.load(response)
    assert "body" not in status
    assert status["camera"] == "head" and status["has_image"]
    assert "VALID" in status["status"] and status["status_age"] >= 0
    with urlopen(base + "/image.jpg?t=1", timeout=3) as response:
        assert response.headers["Content-Type"] == "image/jpeg"
        assert response.read().startswith(b"\xff\xd8")
    raw = np.array([[0, 100, 180, 250, 380, 500]], dtype=np.uint16)
    filtered, meters, keep = module.filter_depth(raw, "16UC1", 0.18, 0.381)
    assert np.array_equal(filtered, [[0, 0, 180, 250, 380, 0]])
    assert np.array_equal(raw, [[0, 100, 180, 250, 380, 500]])
    floating = np.array([[float("nan"), float("inf"), -0.1, 0.25, 0.5]], dtype=np.float32)
    result, _, _ = module.filter_depth(floating, "32FC1", 0.18, 0.38)
    assert np.array_equal(result, [[0, 0, 0, 0.25, 0]])
    request = Request(
        base + "/filter",
        data=json.dumps({"min_m": 0.18, "max_m": 0.38}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=3) as response:
        assert json.load(response)["ok"]
    viewer.depth["left"]["received"] = None
    numeric = bridge.cv2_to_imgmsg(raw, "16UC1")
    viewer.depth_image("left", numeric)
    with urlopen(base + "/depth/left_filtered.jpg", timeout=3) as response:
        assert response.read().startswith(b"\xff\xd8")
    request = Request(
        base + "/filter",
        data=b'{"min_m": 1, "max_m": 0.5}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urlopen(request, timeout=3)
        raise AssertionError("Invalid range must fail")
    except HTTPError as error:
        assert error.code == 400
    assert viewer.filter_near == 0.18 and viewer.filter_far == 0.38
    # The configured jaw-gap mask must remove numeric pixels before detection/publication.
    original_candidate_frame = viewer.candidate_frame
    masked_inputs = []
    viewer.candidate_frame = lambda meters, keep, *args: masked_inputs.append(keep.copy())
    viewer.depth["left"]["received"] = None
    viewer.depth_image("left", bridge.cv2_to_imgmsg(np.full((270, 480), 250, np.uint16), "16UC1"))
    assert not masked_inputs[-1][130, 40]
    assert not masked_inputs[-1][130, 100]
    assert masked_inputs[-1][130, 120] and masked_inputs[-1][130, 300]
    assert masked_inputs[-1][50, 40]
    viewer.candidate_frame = original_candidate_frame
    viewer.exclude_rect = [0.0, 0.0, 0.01, 0.01]
    # A round metric-sized component is selected; clipped background is excluded.
    import cv2

    depth = np.full((240, 320), 0.6, dtype=np.float32)
    cv2.circle(depth, (160, 120), 32, 0.3, -1)
    depth[:70, :100] = 0.3
    keep = (depth >= 0.18) & (depth <= 0.38)
    candidates = module.cup_candidates(depth, keep, (200, 200, 160, 120))
    assert len(candidates) == 1 and candidates[0]["pixel"] == [160, 120]
    assert not module.cup_candidates(depth, np.zeros_like(keep), (200, 200, 160, 120))
    # An adjacent farther table connects in pixels, but must split at its depth edge.
    joined = depth.copy()
    joined[150:, 140:181] = 0.37
    separated = module.cup_candidates(
        joined, (joined >= 0.18) & (joined <= 0.38), (200, 200, 160, 120)
    )
    assert len(separated) == 1 and abs(separated[0]["xyz"][2] - 0.3) < 1e-6
    info = CameraInfo()
    info.width, info.height = 320, 240
    info.header.frame_id = "depth_optical"
    info.p = [200.0, 0.0, 160.0, 0.0, 0.0, 200.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    viewer.camera_info(info)
    message = bridge.cv2_to_imgmsg(depth, "32FC1")
    message.header.frame_id = info.header.frame_id
    for stamp in range(1, 4):
        viewer.depth["left"]["received"] = None
        message.header.stamp.sec = stamp
        viewer.depth_image("left", message)
    state = viewer.state()["candidate"]
    assert state["count"] == 1 and state["stable_frames"] == 3
    assert abs(state["selected"]["xyz"][2] - 0.3) < 1e-6
    with urlopen(base + "/depth/left_candidate.jpg", timeout=3) as response:
        assert response.read().startswith(b"\xff\xd8")
    # Two equal regions are ambiguous, rather than selecting by area or order.
    depth[:] = 0.6
    cv2.circle(depth, (65, 120), 30, 0.3, -1)
    cv2.circle(depth, (255, 120), 30, 0.3, -1)
    message = bridge.cv2_to_imgmsg(depth, "32FC1")
    message.header.frame_id = info.header.frame_id
    viewer.depth["left"]["received"] = None
    viewer.depth_image("left", message)
    state = viewer.state()["candidate"]
    assert state["count"] == 2 and state["selected"] is None and state["stable_frames"] == 0
    viewer.set_filter(0.2, 0.4)
    assert viewer.state()["candidate"]["has_image"] is False
    print("PASS: viewer HTTP RGB/depth endpoints, waiting and numeric/color depth")
finally:
    server.shutdown()
    server.server_close()
    thread.join()
    viewer.destroy_node()
    rclpy.shutdown()
