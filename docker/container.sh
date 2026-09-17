#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTAINER_NAME="cyclo_solution"
DEFAULT_IMAGE="robotis/cyclo-solution:0.1.0"
IMAGE_NAME="${CYCLO_IMAGE:-${DEFAULT_IMAGE}}"
GITHUB_RELEASES_API="https://api.github.com/repos/ROBOTIS-GIT/cyclo_solution/releases/latest"
VERSION_PACKAGE_XML="${SCRIPT_DIR}/../cyclo_cumotion/cyclo_cumotion_bringup/package.xml"

# Function to display help
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

get_current_version() {
    local version=""
    if [ -f "${VERSION_PACKAGE_XML}" ]; then
        version=$(sed -n \
            's/.*<version>\([^<]*\)<\/version>.*/\1/p' \
            "${VERSION_PACKAGE_XML}" | head -1)
    fi
    echo "${version:-unknown}"
}

get_latest_version() {
    local response tag
    response=$(curl -sL --connect-timeout 5 "${GITHUB_RELEASES_API}" 2>/dev/null)
    tag=$(echo "${response}" | sed -n \
        's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
    echo "${tag#v}"
}

update_available() {
    local current_version="$1"
    local latest_version="$2"
    local newer_version

    if [ -z "${latest_version}" ] || [ "${latest_version}" = "${current_version}" ]; then
        return 1
    fi
    newer_version=$(printf '%s\n%s\n' \
        "${current_version}" "${latest_version}" | sort -V | tail -1)
    [ "${newer_version}" = "${latest_version}" ]
}

print_update_notice() {
    local current_version="$1"
    local latest_version="$2"

    echo ""
    echo "New Cyclo Solution release available: ${latest_version} (current: ${current_version})"
    echo "Update the repository, then restart the container to use the new release."
    echo ""
}

check_for_update() {
    local current_version latest_version

    current_version=$(get_current_version)
    latest_version=$(get_latest_version)
    if update_available "${current_version}" "${latest_version}"; then
        print_update_notice "${current_version}" "${latest_version}"
    fi
}

# Function to start the container
start_container() {
    # Set up X11 forwarding only if DISPLAY is set
    if [ -n "$DISPLAY" ]; then
        echo "Setting up X11 forwarding..."
        xhost +local:docker || true
    else
        echo "Warning: DISPLAY environment variable is not set. X11 forwarding will not be available."
    fi

    case "$(uname -m)" in
        x86_64|amd64) ;;
        *) echo "Error: This environment currently supports x86_64 only."; return 1 ;;
    esac

    echo "Starting cyclo_solution with ${IMAGE_NAME}..."

    if [ "${CYCLO_SKIP_PULL:-0}" != "1" ] && [ "${IMAGE_NAME}" = "${DEFAULT_IMAGE}" ]; then
        check_for_update
    fi

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

# Function to enter the container
enter_container() {
    # Set up X11 forwarding only if DISPLAY is set
    if [ -n "$DISPLAY" ]; then
        echo "Setting up X11 forwarding..."
        xhost +local:docker || true
    else
        echo "Warning: DISPLAY environment variable is not set. X11 forwarding will not be available."
    fi

    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        echo "Error: Container is not running"
        return 1
    fi

    if [ "${CYCLO_IMAGE:-${DEFAULT_IMAGE}}" = "${DEFAULT_IMAGE}" ]; then
        check_for_update
    fi

    docker exec -it "${CONTAINER_NAME}" bash
}

# Function to stop the container
stop_container() {
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        echo "Error: Container is not running"
        return 1
    fi

    echo "Warning: This will stop and remove the container. All unsaved data in the container will be lost."
    read -p "Are you sure you want to continue? [y/N] " -n 1 -r
    echo
    if [[ ${REPLY} =~ ^[Yy]$ ]]; then
        docker compose -f "${SCRIPT_DIR}/docker-compose.yml" down
    else
        echo "Operation cancelled."
    fi
}

# Main command handling
case "$1" in
    "help")
        show_help
        ;;
    "start")
        start_container
        ;;
    "enter")
        enter_container
        ;;
    "stop")
        stop_container
        ;;
    *)
        echo "Error: Unknown command"
        show_help
        exit 1
        ;;
esac
