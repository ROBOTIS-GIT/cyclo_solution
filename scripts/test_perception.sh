#!/usr/bin/env bash
# Run one manual camera test session inside the existing demo container.
set -euo pipefail

task_container="${DEMO_CONTAINER:-docker-demo_1f-1}"
if [[ "$(docker inspect -f '{{.State.Running}}' "$task_container" 2>/dev/null || true)" != true ]]; then
    printf 'Demo container %s is not running. Start it with docker/container.sh start first.\n' \
        "$task_container" >&2
    exit 1
fi
task_flags=(-i)
if [[ -t 0 && -t 1 ]]; then task_flags=(-it); fi
exec docker exec "${task_flags[@]}" "$task_container" \
    /root/ros2_ws/src/demo_1f/docker/entrypoint.sh \
    ros2 launch demo_1f perception_test.launch.py \
    host:=0.0.0.0 profiles:=/demo_config/perception_test.yaml "$@"
