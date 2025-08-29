#!/bin/bash
# launch_multi_gpu.sh - Launch multiple Orpheus TTS instances across GPUs

set -e

# Load configuration
SCRIPT_DIR="$(dirname "$0")"
if [ -f "$SCRIPT_DIR/multi_gpu.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/multi_gpu.env" | xargs)
fi

# Default values if not set
BASE_PORT=${BASE_PORT:-9090}
NUM_GPUS=${NUM_GPUS:-8}
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/logs"}
PID_DIR=${PID_DIR:-"$SCRIPT_DIR/pids"}

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create directories if they don't exist
mkdir -p "$LOG_DIR" "$PID_DIR"

# Function to start a single instance
start_instance() {
    local gpu_id=$1
    local port=$((BASE_PORT + gpu_id))
    local log_file="$LOG_DIR/orpheus_gpu${gpu_id}.log"
    local pid_file="$PID_DIR/orpheus_gpu${gpu_id}.pid"
    
    echo -e "${GREEN}Starting Orpheus on GPU $gpu_id (port $port)...${NC}"
    
    # Set environment variables for this instance
    export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
    export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
    export WORLD_SIZE=1
    export RANK=0
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export SERVER_PORT=$port
    
    # Disable MPI and NCCL for single GPU operation
    export TRTLLM_DISABLE_MPI=1
    export TENSORRT_LLM_USE_MPI=0
    export TRTLLM_SINGLE_WORKER=1
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    
    # Pass through model and TRT configuration
    export MODEL_NAME=${MODEL_NAME:-"canopylabs/orpheus-3b-0.1-ft"}
    export TRT_DTYPE=${TRT_DTYPE:-"bfloat16"}
    export TRT_MAX_BATCH_SIZE=${TRT_MAX_BATCH_SIZE:-4}
    export TRT_MAX_INPUT_LEN=${TRT_MAX_INPUT_LEN:-1024}
    export TRT_MAX_SEQ_LEN=${TRT_MAX_SEQ_LEN:-8192}
    export TRT_MAX_TOKENS=${TRT_MAX_TOKENS:-1200}
    export TRT_FREE_GPU_MEMORY_FRACTION=${TRT_FREE_GPU_MEMORY_FRACTION:-0.6}
    export TRT_TEMPERATURE=${TRT_TEMPERATURE:-0.1}
    export TRT_TOP_P=${TRT_TOP_P:-0.95}
    export TRT_REPETITION_PENALTY=${TRT_REPETITION_PENALTY:-1.1}
    export TRT_STOP_TOKEN_IDS=${TRT_STOP_TOKEN_IDS:-"128258"}
    
    # Start the server in the background (from parent directory)
    cd "$SCRIPT_DIR/.."
    
    # Add conda lib path for tensorrt_llm
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    
    nohup uvicorn main:app --host 0.0.0.0 --port $port > "$log_file" 2>&1 &
    
    # Save PID
    echo $! > "$pid_file"
    
    echo -e "${GREEN}Started instance on GPU $gpu_id (PID: $(cat $pid_file))${NC}"
}

# Check if instances are already running
check_running() {
    local running=0
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        local pid_file="$PID_DIR/orpheus_gpu${gpu_id}.pid"
        if [ -f "$pid_file" ] && kill -0 $(cat "$pid_file") 2>/dev/null; then
            running=$((running + 1))
        fi
    done
    return $running
}

# Stop any existing instances
echo "Checking for existing instances..."
check_running
if [ $? -gt 0 ]; then
    echo -e "${RED}Found running instances. Please stop them first with: ./stop_multi_gpu.sh${NC}"
    exit 1
fi

# Start instances on each GPU
echo "Starting $NUM_GPUS Orpheus instances..."

# Start first instance and wait for it to initialize (download model/build engine)
echo -e "\n${YELLOW}Starting first instance to download model and build TRT engine...${NC}"
start_instance 0

# Wait for first instance to be ready
echo -e "${YELLOW}Waiting for first instance to initialize (this may take 5-15 minutes on first run)...${NC}"
port=$BASE_PORT
max_wait=900  # 15 minutes max
elapsed=0
while [ $elapsed -lt $max_wait ]; do
    if curl -s -o /dev/null -w '' "http://localhost:$port/docs" 2>/dev/null; then
        echo -e "\n${GREEN}First instance is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ $elapsed -ge $max_wait ]; then
    echo -e "\n${RED}First instance failed to start after 15 minutes${NC}"
    echo "Check logs: tail -f $LOG_DIR/orpheus_gpu0.log"
    exit 1
fi

# Now start remaining instances (they'll reuse the cached model/engine)
echo -e "\n${GREEN}Starting remaining instances (using cached model/engine)...${NC}"
for gpu_id in $(seq 1 $((NUM_GPUS - 1))); do
    start_instance $gpu_id
    sleep 2  # Small delay between starts
done

echo -e "\n${GREEN}All instances started!${NC}"
echo "Instances are running on ports ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
echo "Logs are in: $LOG_DIR/"
echo "To stop all instances: ./stop_multi_gpu.sh"
