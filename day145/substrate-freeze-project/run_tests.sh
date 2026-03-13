#!/bin/bash
# Runs verification against the running service (must be started via start.sh or setup.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="8080"
BASE_URL="http://localhost:${PORT}"

if ! curl -s --connect-timeout 2 "${BASE_URL}/status" > /dev/null; then
    echo "Service not running on port ${PORT}. Start it with: ${SCRIPT_DIR}/start.sh (native) or run setup.sh --docker"
    exit 1
fi

# Ensure service is unfrozen so /set can run
curl -s -X POST "${BASE_URL}/unfreeze" > /dev/null

echo "Running tests..."
curl -s "${BASE_URL}/status" | grep -q "running\|frozen" && echo "  [OK] /status" || { echo "  [FAIL] /status"; exit 1; }
curl -s -X POST "${BASE_URL}/set?key=test_key&value=test_val" | grep -q "Set key" && echo "  [OK] /set" || { echo "  [FAIL] /set"; exit 1; }
curl -s "${BASE_URL}/get?key=test_key" | grep -q "test_val" && echo "  [OK] /get" || { echo "  [FAIL] /get"; exit 1; }
curl -s -X POST "${BASE_URL}/freeze" | grep -q "FROZEN\|frozen" && echo "  [OK] /freeze" || { echo "  [FAIL] /freeze"; exit 1; }
curl -s "${BASE_URL}/dump" | grep -q "test_key" && echo "  [OK] /dump" || { echo "  [FAIL] /dump"; exit 1; }
curl -s -X POST "${BASE_URL}/unfreeze" | grep -q "UNFROZEN\|unfreeze" && echo "  [OK] /unfreeze" || { echo "  [FAIL] /unfreeze"; exit 1; }
echo "All tests passed."
