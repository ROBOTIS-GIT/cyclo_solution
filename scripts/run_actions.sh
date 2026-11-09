#!/usr/bin/env bash
# Host-side interactive client inside the already running demo container.
set -euo pipefail
task_container="${DEMO_CONTAINER:-docker-demo_1f-1}"
if [[ "$(docker inspect -f '{{.State.Running}}' "$task_container" 2>/dev/null || true)" != true ]]; then
    printf 'Start the demo container with docker/container.sh start first.\n' >&2
    exit 1
fi
task_flags=(-i)
if [[ -t 0 && -t 1 ]]; then task_flags=(-it); fi
exec docker exec "${task_flags[@]}" "$task_container" \
    /root/ros2_ws/src/demo_1f/docker/entrypoint.sh \
    ros2 run demo_1f action_runner.py "$@"
