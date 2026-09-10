#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="cyclo_solution"
IMAGE_NAME="robotis/cyclo-solution:0.1.0-local"
CACHE_IMAGE="robotis/cyclo-solution:4.6-local"

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
    echo "  start                   Start the container"
    echo "  enter                   Enter the running container"
    echo "  stop                    Stop the container"
    echo ""
    echo "Examples:"
    echo "  $0 start                Build and start the container"
    echo "  $0 enter                Enter the running container"
    echo "  $0 stop                 Stop the container"
}

start_container() {
    configure_x11

    case "$(uname -m)" in
        x86_64|amd64) ;;
        *) echo "Error: This environment currently supports x86_64 only."; return 1 ;;
    esac

    echo "Building ${IMAGE_NAME}..."
    BUILD_ARGS=()
    if docker image inspect "${CACHE_IMAGE}" >/dev/null 2>&1; then
        echo "Reusing dependency layers from ${CACHE_IMAGE}."
        BUILD_ARGS+=(
            --build-arg "BASE_IMAGE=${CACHE_IMAGE}"
            --build-arg "USE_PREBUILT_DEPENDENCIES=1"
        )
    fi
    docker build \
        "${BUILD_ARGS[@]}" \
        -f "${SCRIPT_DIR}/Dockerfile.amd64" \
        -t "${IMAGE_NAME}" \
        "${SCRIPT_DIR}/.." || return 1

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
