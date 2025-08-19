#!/bin/bash
set -e

# Change to parent directory
cd "$(dirname "$0")/.."

# Run the Docker container
echo "Starting Orpheus TTS Server..."

# Create local directories for volume mounts
mkdir -p logs outputs

# Run with GPU support
docker run \
  --gpus all \
  --name orpheus-tts \
  --rm \
  --env-file .env \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HUB_OFFLINE=0 \
  -p 9090:9090 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/outputs:/app/outputs \
  -v orpheus_hf_cache:/workspace/hf \
  orpheus-tts:latest

# Note: Remove --rm flag if you want the container to persist after stopping
