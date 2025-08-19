# Orpheus TTS Server - Quick Start Guide

This guide helps you get the Orpheus TTS Server running quickly with Docker.

## Prerequisites Checklist

- [ ] NVIDIA GPU with CUDA support
- [ ] Docker installed (`docker --version`)
- [ ] NVIDIA Container Toolkit installed
- [ ] AWS CLI installed (for ECR option)

## Choose Your Deployment Path

### 🚀 Path 1: Use Pre-built Image (Fastest)

**Time to deploy: ~5 minutes**

1. **Create .env file** in your working directory:
```bash
cat > .env << 'EOF'
HF_HOME=/workspace/hf
TRANSFORMERS_OFFLINE=0
HF_HUB_OFFLINE=0
MODEL_NAME=TrySalient/tts-collections-test-verbalized
LOG_LEVEL=INFO
AVAILABLE_VOICES=tara,zoe,jess,zac,leo,mia,julia,leah
TRT_TEMPERATURE=0.1
TRT_TOP_P=0.95
TRT_MAX_TOKENS=1200
TRT_REPETITION_PENALTY=1.1
TRT_STOP_TOKEN_IDS=128258
TRT_DTYPE=bfloat16
TRT_MAX_INPUT_LEN=1024
TRT_MAX_BATCH_SIZE=16
TRT_MAX_SEQ_LEN=2048
TRT_ENABLE_CHUNKED_PREFILL=True
TRT_MAX_BEAM_WIDTH=1
TRT_MAX_NUM_TOKENS=16384
TRT_FREE_GPU_MEMORY_FRACTION=0.85
TRT_KV_CACHE_MAX_TOKENS=512
EOF
```

2. **Authenticate with ECR**:
```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  308804103509.dkr.ecr.us-west-2.amazonaws.com
```

3. **Run the server**:
```bash
docker run --gpus all \
  --name orpheus-tts \
  --rm \
  --env-file .env \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HUB_OFFLINE=0 \
  -p 9090:9090 \
  -v orpheus_hf_cache:/workspace/hf \
  308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:latest
```

### 🔧 Path 2: Build from Source

**Time to deploy: ~15-20 minutes**

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/orpheus-streaming.git
cd orpheus-streaming
```

2. **Copy and configure .env**:
```bash
# Copy the example from the deployment folder
cp deployment/.env.example .env
# Edit as needed
nano .env
```

3. **Build and run**:
```bash
# Build the image
./deployment/docker-build.sh

# Run the server
./deployment/docker-run.sh
```

## Verify It's Working

1. **Check server health**:
```bash
# Should see logs showing model loading
docker logs orpheus-tts

# Wait for "Application startup complete" message
```

2. **Test the API**:
```bash
# List voices
curl http://localhost:9090/api/voices | jq .

# Generate speech
curl -X POST http://localhost:9090/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "tara"}' \
  --output test.pcm

# Convert to WAV (requires ffmpeg)
ffmpeg -f s16le -ar 24000 -ac 1 -i test.pcm test.wav
```

3. **Run performance benchmark**:
```bash
curl -X POST http://localhost:9090/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "voice": "tara",
    "num_runs": 5
  }' | jq .
```

## Expected Performance

- **First startup**: 30-60s (model loading + TRT engine building)
- **TTFB**: ~337ms on A100/RTX 4090
- **RTF**: ~0.92 (faster than real-time)
- **Memory usage**: ~30GB GPU memory

## Common Issues

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Port already in use
```bash
# Use a different port
docker run --gpus all \
  -p 9091:9090 \
  ... (other args)
```

### AWS credentials error (ECR path)
```bash
# Configure AWS credentials
aws configure
# Or use temporary credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

## Stop the Server

```bash
# Stop container
docker stop orpheus-tts

# Remove container and volumes (clean slate)
docker rm orpheus-tts
docker volume rm orpheus_hf_cache
```

## Next Steps

- See [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) for detailed configuration
- See [ECR_DEPLOYMENT.md](./ECR_DEPLOYMENT.md) for ECR-specific details
- Check the main [README.md](../README.md) for API documentation
