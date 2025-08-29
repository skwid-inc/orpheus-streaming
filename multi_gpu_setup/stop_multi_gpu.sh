#!/bin/bash
# stop_multi_gpu.sh - Stop all Orpheus TTS instances

set -e

# Configuration
SCRIPT_DIR="$(dirname "$0")"
NUM_GPUS=8
PID_DIR="$SCRIPT_DIR/pids"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "Stopping all Orpheus instances..."

stopped=0
not_running=0

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    pid_file="$PID_DIR/orpheus_gpu${gpu_id}.pid"
    
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            echo -e "${YELLOW}Stopping instance on GPU $gpu_id (PID: $pid)...${NC}"
            kill $pid
            
            # Wait for process to stop
            timeout=10
            while kill -0 $pid 2>/dev/null && [ $timeout -gt 0 ]; do
                sleep 1
                timeout=$((timeout - 1))
            done
            
            if kill -0 $pid 2>/dev/null; then
                echo -e "${RED}Force killing instance on GPU $gpu_id...${NC}"
                kill -9 $pid
            fi
            
            echo -e "${GREEN}Stopped instance on GPU $gpu_id${NC}"
            stopped=$((stopped + 1))
        else
            echo -e "${YELLOW}Instance on GPU $gpu_id was not running${NC}"
            not_running=$((not_running + 1))
        fi
        rm -f "$pid_file"
    else
        not_running=$((not_running + 1))
    fi
done

echo -e "\n${GREEN}Summary:${NC}"
echo "- Stopped: $stopped instances"
echo "- Not running: $not_running instances"
