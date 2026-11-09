#!/usr/bin/env bash
set -euo pipefail
task_demo_dir="$(cd "$(dirname "$0")/.." && pwd)"
task_robot_dir="${1:-$(dirname "$task_demo_dir")/ai_worker}"
docker run --rm --network none \
  -e DEMO_ISOLATED_TEST=1 -e RMW_IMPLEMENTATION=rmw_zenoh_cpp -e ROS_DOMAIN_ID=30 \
  --entrypoint /bin/bash \
  -v "$task_demo_dir:/root/ros2_ws/src/demo_1f" -v "$task_robot_dir:/robot:ro" \
  -w /root/ros2_ws "${DEMO_BASE_IMAGE:-robotis/ai-worker:2.2.7}" -c '
set -e
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
colcon build --packages-up-to demo_1f \
  --base-paths /root/ros2_ws/src/demo_1f /root/ros2_ws/src/demo_1f/modules/dynamixel_pick/ffw_* \
  --cmake-args -DBUILD_TESTING=ON
source /root/ros2_ws/install/setup.bash
ros2 run rmw_zenoh_cpp rmw_zenohd >/tmp/demo_zenoh.log 2>&1 &
router_pid=$!
trap "kill $router_pid 2>/dev/null || true" EXIT
colcon test --packages-select demo_1f \
  --event-handlers console_direct+ --return-code-on-test-failure
# This offline geometry tool does not communicate with other nodes. The base
# MoveIt library has a Zenoh logger teardown bug; use FastDDS only for this tool.
RMW_IMPLEMENTATION=rmw_fastrtps_cpp /root/ros2_ws/build/demo_1f/validate_sg2_targets \
  /root/ros2_ws/install/demo_1f/share/demo_1f/urdf/sg2_extended_grippers.urdf \
  /root/ros2_ws/src/demo_1f/config/sg2_handover.srdf \
  --ros-args --params-file /root/ros2_ws/src/demo_1f/config/handover.yaml
python3 /root/ros2_ws/src/demo_1f/test/perception_viewer_smoke.py
python3 /root/ros2_ws/src/demo_1f/test/test_gripper_model.py
python3 /root/ros2_ws/src/demo_1f/test/launch_smoke.py
python3 /root/ros2_ws/src/demo_1f/test/sg2_integration.py \
  --urdf /root/ros2_ws/install/demo_1f/share/demo_1f/urdf/sg2_extended_grippers.urdf \
  --approach --report /root/ros2_ws/log/motion_smooth.json
'
