#!/bin/bash

# Startup script from sources directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting from sources directory..."
cd "$PROJECT_DIR"

if [ -f "startup.sh" ]; then
    bash "$PROJECT_DIR/startup.sh"
else
    echo "Error: startup.sh not found in project directory"
    exit 1
fi
