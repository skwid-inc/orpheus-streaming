## 🚀 Deployment Options

### Option 1: Use Pre-built Image from ECR (Fastest - ~5 minutes)

1. **Create .env file** in project root:
```bash
cat > ../.env << 'EOF'
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
# HF_TOKEN=hf_xxxx  # Add yours
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
  --env-file ../.env \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HUB_OFFLINE=0 \
  -p 9090:9090 \
  -v orpheus_hf_cache:/workspace/hf \
  308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:latest
```

### Option 2: Build from Source (~15 minutes)

From the project root:

1. **Build the image**:
```bash
./deployment/docker-build.sh
```

2. **Run the server**:
```bash
./deployment/docker-run.sh
```

Or use Docker Compose:
```bash
cd deployment && docker compose up -d
```

## 🧪 Testing the Deployment

### Quick Test
```bash
# Check available voices
curl http://localhost:9090/api/voices | jq .

# Generate speech
curl -X POST http://localhost:9090/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "tara", "model": "tts-1"}' \
  --output test.wav
```



### Benchmark Performance
```bash
curl -X POST http://localhost:9090/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a benchmark test.",
    "voice": "tara",
    "num_runs": 5,
    "warmup": true
  }' | jq .
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

- `MODEL_NAME`: HuggingFace model to use (default: TrySalient/tts-collections-test-verbalized)
- `HF_TOKEN`: (Optional) HuggingFace token for private models
- `AVAILABLE_VOICES`: Comma-separated list of voice names
- `TRT_*`: TensorRT-LLM configuration parameters
- `LOG_LEVEL`: Logging verbosity (INFO, DEBUG, etc.)

### Hardware Requirements

Use A100

## 🐛 Troubleshooting

### Container won't start
- Check GPU availability: `nvidia-smi`
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### Model download fails
- Ensure `TRANSFORMERS_OFFLINE=0` and `HF_HUB_OFFLINE=0` are set
- Check internet connectivity
- For private models, verify `HF_TOKEN` is set correctly

### Performance issues
- First startup compiles TRT engine (can take 5-10 minutes)
- Subsequent runs use cached engine
- Check benchmark results with `/v1/benchmark` endpoint

## 🔒 Security Notes

- Never commit `.env` file with sensitive tokens
- Use volume mounts for model cache to avoid re-downloads
- Consider network isolation for production deployments

## 📝 Additional Notes

- The Docker image is ~34.5GB (includes CUDA, PyTorch, TensorRT)
- First run downloads models (~10GB) to the HuggingFace cache
- TRT engine compilation happens on first startup
- Use `docker compose` (not `docker-compose`) for newer Docker versions