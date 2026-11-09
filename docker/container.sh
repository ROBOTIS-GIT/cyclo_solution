#!/usr/bin/env bash
set -euo pipefail

task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_compose_file="${task_script_dir}/compose.yaml"
task_service="demo_1f"

show_help() {
    cat <<EOF
Usage: $0 [command]

Commands:
  help     Show this help message
  start    Build the image and start the idle container
  enter    Enter the running container with the ROS environment loaded
  stop     Stop the container without removing it

Examples:
  $0 start
  $0 enter
  $0 stop

Inside the container, start the demo manually:
  ros2 launch demo_1f full_demo.launch.py use_sim_time:=false
EOF
}

compose() {
    docker compose -f "$task_compose_file" "$@"
}

if (( $# > 1 )); then
    show_help >&2
    exit 2
fi

task_command="${1:-help}"
case "$task_command" in
    help|-h|--help)
        show_help
        exit 0
        ;;
    start|enter|stop) ;;
    *)
        printf 'Error: Unknown command: %s\n' "$task_command" >&2
        show_help >&2
        exit 2
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    printf 'Error: Docker is not installed or is not on PATH.\n' >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    printf 'Error: Docker Compose v2 (docker compose) is required.\n' >&2
    exit 1
fi

case "$task_command" in
    start)
        compose up -d --build "$task_service"
        printf 'Container is ready. The demo and MoveIt must be started manually.\n'
        printf 'Enter with: %s enter\n' "$0"
        ;;
    enter)
        task_container_id="$(compose ps --status running --quiet "$task_service")"
        if [[ -z "$task_container_id" ]]; then
            printf 'Error: demo_1f is not running. Run: %s start\n' "$0" >&2
            exit 1
        fi
        exec docker compose -f "$task_compose_file" exec "$task_service" bash
        ;;
    stop)
        compose stop "$task_service"
        ;;
esac
