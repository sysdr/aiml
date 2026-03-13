#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="substrate-freeze-project"
BINARY_NAME="substrate-freeze-app"
PORT="8080"

cd "${SCRIPT_DIR}/${PROJECT_DIR}" || exit 1
if [ ! -f "./${BINARY_NAME}" ]; then
    echo "Binary not found. Run setup.sh first."
    exit 1
fi
./"${BINARY_NAME}" &
echo $! > "${BINARY_NAME}.pid"
echo "Service started on http://localhost:${PORT} (PID: $(cat ${BINARY_NAME}.pid)). Use stop.sh to stop."
