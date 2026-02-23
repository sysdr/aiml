#!/bin/bash

echo "🐳 Building Docker containers..."

# Build and start with Docker Compose
docker-compose up --build -d

echo "✅ Docker containers started!"
echo "🌐 Application: http://localhost:3000"
echo "📡 API: http://localhost:8000"

# Show logs
docker-compose logs -f
