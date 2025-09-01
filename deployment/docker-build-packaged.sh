#!/bin/bash

# Build script for packaged Orpheus TTS image
# This script builds the image with the model and nginx config included

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Orpheus TTS with packaged model...${NC}"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set. You may need to set it if the model is private:${NC}"
    echo "export HF_TOKEN=your_huggingface_token_here"
    echo ""
    echo -e "${YELLOW}Continuing without token...${NC}"
fi

# Build the image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    --build-arg HF_TOKEN="${HF_TOKEN}" \
    -f "deployment copy/Dockerfile.packaged" \
    -t orpheus-tts:latest \
    .

echo -e "${GREEN}Build complete!${NC}"
echo ""
echo -e "${GREEN}Image details:${NC}"
docker images orpheus-tts:latest

echo ""
echo -e "${GREEN}To run the packaged deployment:${NC}"
echo "docker-compose -f docker-compose-packaged.yml up -d"

echo ""
echo -e "${GREEN}Benefits of this packaged image:${NC}"
echo "✅ Model is pre-downloaded and cached in the image"
echo "✅ No HuggingFace downloads needed at runtime"
echo "✅ Nginx config is included in the image"
echo "✅ Faster startup times in customer environments"
echo "✅ Works in air-gapped environments"
