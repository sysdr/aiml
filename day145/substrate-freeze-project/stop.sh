#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="substrate-freeze-project"
BINARY_NAME="substrate-freeze-app"
DOCKER_CONTAINER_NAME="substrate-freeze-container"

if [[ "$1" == "--docker" ]]; then
    docker stop "${DOCKER_CONTAINER_NAME}" 2>/dev/null && docker rm "${DOCKER_CONTAINER_NAME}" 2>/dev/null && echo "Docker container stopped."
else
    if [ -f "${SCRIPT_DIR}/${PROJECT_DIR}/${BINARY_NAME}.pid" ]; then
        PID=$(cat "${SCRIPT_DIR}/${PROJECT_DIR}/${BINARY_NAME}.pid")
        kill "${PID}" 2>/dev/null && echo "Stopped native process ${PID}."
        rm -f "${SCRIPT_DIR}/${PROJECT_DIR}/${BINARY_NAME}.pid"
    fi
    pkill -f "./${BINARY_NAME}" 2>/dev/null
    echo "Native service stopped."
fi
