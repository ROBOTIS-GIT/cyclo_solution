#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="cyclo_solution"
DEFAULT_IMAGE="robotis/cyclo-solution:0.1.0"
IMAGE_NAME="${CYCLO_IMAGE:-${DEFAULT_IMAGE}}"

configure_x11() {
    if [ -z "${DISPLAY:-}" ]; then
        echo "Warning: DISPLAY is not set. GUI applications will not be available."
        return 0
    fi

    if ! command -v xhost >/dev/null 2>&1; then
        echo "Warning: xhost is not installed on the host. GUI authorization was not configured."
        return 0
    fi

    # The container runs as root. Reapply this permission for every host login
    # session because Xwayland authentication is regenerated after logout/reboot,
    # while Docker may restart the container automatically.
    if ! xhost +si:localuser:root >/dev/null; then
        echo "Warning: Failed to authorize container GUI access for DISPLAY=${DISPLAY}."
    fi
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  help                    Show this help message"
    echo "  start                   Pull and start the container"
    echo "  enter                   Enter the running container"
    echo "  stop                    Stop the container"
    echo ""
    echo "Examples:"
    echo "  $0 start                Pull and start the released image"
    echo "  $0 enter                Enter the running container"
    echo "  $0 stop                 Stop the container"
    echo ""
    echo "Local image override:"
    echo "  CYCLO_IMAGE=<tag> CYCLO_SKIP_PULL=1 $0 start"
}

start_container() {
    configure_x11

    case "$(uname -m)" in
        x86_64|amd64) ;;
        *) echo "Error: This environment currently supports x86_64 only."; return 1 ;;
    esac

    echo "Starting cyclo_solution with ${IMAGE_NAME}..."

    if [ "${CYCLO_SKIP_PULL:-0}" = "1" ]; then
        if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
            echo "Error: Local image '${IMAGE_NAME}' was not found."
            return 1
        fi
        echo "Using local image without pulling."
    else
        CYCLO_IMAGE="${IMAGE_NAME}" \
            docker compose -f "${SCRIPT_DIR}/docker-compose.yml" pull || return 1
    fi

    CYCLO_IMAGE="${IMAGE_NAME}" \
        docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d
}

enter_container() {
    configure_x11

    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        echo "Error: Container is not running"
        return 1
    fi

    docker exec -it "${CONTAINER_NAME}" bash
}

stop_container() {
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        echo "Error: Container is not running"
        return 1
    fi

    echo "Warning: This will stop and remove the container."
    read -p "Are you sure you want to continue? [y/N] " -n 1 -r
    echo
    if [[ ${REPLY} =~ ^[Yy]$ ]]; then
        docker compose -f "${SCRIPT_DIR}/docker-compose.yml" down
    else
        echo "Operation cancelled."
    fi
}

case "${1:-}" in
    help) show_help ;;
    start) start_container ;;
    enter) enter_container ;;
    stop) stop_container ;;
    *)
        echo "Error: Unknown command"
        show_help
        exit 1
        ;;
esac
