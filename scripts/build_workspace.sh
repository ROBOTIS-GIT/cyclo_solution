#!/usr/bin/env bash
# Run inside the existing ROS container; no image build or robot launch.
set -eo pipefail
task_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_workspace="${DEMO_WORKSPACE:-/root/ros2_ws}"
source /opt/ros/jazzy/setup.bash
if [[ -f /root/ros2_ws/install/setup.bash ]]; then
  source /root/ros2_ws/install/setup.bash
fi
mkdir -p "$task_workspace"
cd "$task_workspace"
# Explicit paths discover sibling packages vendored beneath a ROS package root.
colcon build --symlink-install \
  --base-paths "$task_source" "$task_source"/modules/dynamixel_pick/ffw_* \
  --packages-up-to demo_1f \
  --parallel-workers 2 --cmake-args -DBUILD_TESTING="${BUILD_TESTING:-OFF}"
