#!/bin/bash
# check_cache.sh - Check if model and TRT engine are cached

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load configuration
SCRIPT_DIR="$(dirname "$0")"
if [ -f "$SCRIPT_DIR/multi_gpu.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/multi_gpu.env" | xargs)
fi

MODEL_NAME=${MODEL_NAME:-"canopylabs/orpheus-3b-0.1-ft"}
HF_HOME=${HF_HOME:-"/workspace/hf"}

echo -e "${BLUE}=== Orpheus Cache Status ===${NC}\n"

# Check HuggingFace cache
echo -e "${BLUE}HuggingFace Model Cache:${NC}"
MODEL_PATH="$HF_HOME/hub/models--${MODEL_NAME//\/--}"
if [ -d "$MODEL_PATH" ]; then
    SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    echo -e "${GREEN}✓ Model cached${NC} at: $MODEL_PATH (Size: $SIZE)"
else
    echo -e "${RED}✗ Model not cached${NC}"
    echo "  Will download on first run (~10GB)"
fi

# Check TensorRT-LLM engine cache
echo -e "\n${BLUE}TensorRT-LLM Engine Cache:${NC}"
TRT_CACHE_DIR="$HOME/.cache/tensorrt_llm"
if [ -d "$TRT_CACHE_DIR" ]; then
    # Look for engine files
    ENGINE_COUNT=$(find "$TRT_CACHE_DIR" -name "*.engine" 2>/dev/null | wc -l)
    if [ $ENGINE_COUNT -gt 0 ]; then
        SIZE=$(du -sh "$TRT_CACHE_DIR" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓ TRT engines cached${NC} at: $TRT_CACHE_DIR (Size: $SIZE)"
        echo "  Found $ENGINE_COUNT engine file(s)"
    else
        echo -e "${YELLOW}⚠ TRT cache exists but no engines found${NC}"
        echo "  Engines will be built on first run (5-15 minutes)"
    fi
else
    echo -e "${RED}✗ TRT engine not cached${NC}"
    echo "  Will build on first run (5-15 minutes)"
fi

# Estimate initialization time
echo -e "\n${BLUE}Estimated startup time:${NC}"
if [ -d "$MODEL_PATH" ] && [ $ENGINE_COUNT -gt 0 ]; then
    echo -e "${GREEN}< 1 minute${NC} (everything cached)"
elif [ -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}5-15 minutes${NC} (need to build TRT engine)"
else
    echo -e "${RED}15-30 minutes${NC} (need to download model and build engine)"
fi
