#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Building image..."
docker build -t substrate-freeze-app .

docker rm -f substrate-freeze 2>/dev/null || true

echo "Starting server on port 8080..."
docker run -d --rm -p 8080:8080 --name substrate-freeze substrate-freeze-app

echo "Waiting for server to be ready..."
for i in 1 2 3 4 5; do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/status 2>/dev/null | grep -q 200; then
    echo "Server is up."
    echo ""
    echo "  Dashboard:  http://127.0.0.1:8080/dashboard"
    echo "  API:        http://127.0.0.1:8080/"
    echo ""
    echo "If the browser shows Connection Refused, use: http://127.0.0.1:8080/dashboard"
    echo "Stop server:  docker stop substrate-freeze"
    exit 0
  fi
  sleep 1
done

echo "Server may still be starting. Try in a few seconds:"
echo "  http://127.0.0.1:8080/dashboard"
echo ""
echo "Stop server: docker stop substrate-freeze"
