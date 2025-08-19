#!/bin/bash
set -e

# Change to parent directory for correct build context
cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Orpheus TTS Docker image..."
docker build -t orpheus-tts:latest -f deployment/Dockerfile .

echo "Build complete! You can now run the container with:"
echo "  docker run --gpus all -p 9090:9090 orpheus-tts:latest"
echo ""
echo "Or use docker-compose:"
echo "  cd deployment && docker-compose up -d"
