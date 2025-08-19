# Docker Deployment Guide for Orpheus TTS Server

This guide provides instructions for deploying the Orpheus TTS Server using Docker on A100 GPUs.

## Prerequisites

- NVIDIA A100 GPU
- Docker installed with NVIDIA Container Toolkit
- NVIDIA drivers (CUDA 12.4 compatible)
- Your `.env` file configured with necessary parameters

## Quick Start

### 1. Build the Docker Image

From the project root:
```bash
./deployment/docker-build.sh
```

Or manually from the project root:
```bash
docker build -t orpheus-tts:latest -f deployment/Dockerfile .
```

### 2. Run the Server

Using the provided script from the project root:
```bash
./deployment/docker-run.sh
```

Or using docker-compose:
```bash
cd deployment && docker-compose up -d
```

Or manually with docker run from the project root:
```bash
docker run --gpus all --env-file .env -p 9090:9090 orpheus-tts:latest
```

### 3. Test the Endpoints

The server will be available at `http://localhost:9090`. Test it with:

```bash
# Health check
curl http://localhost:9090/health

# List available voices
curl http://localhost:9090/v1/voices

# Generate speech (streaming)
curl -X POST http://localhost:9090/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of the Orpheus TTS system.",
    "voice": "tara"
  }' \
  --output test.pcm

# Run performance benchmark
curl -X POST http://localhost:9090/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test text for benchmarking TTS performance.",
    "voice": "tara",
    "num_runs": 5,
    "warmup": true
  }'
```

You can also use the provided test scripts from the project root:
```bash
# Test deployment
./deployment/test_docker_deployment.sh

# Test benchmark API (create this in the project root)
python test_benchmark_api.py
```

## Docker Commands

### View logs
```bash
docker logs orpheus-tts
```

### Stop the container
```bash
docker stop orpheus-tts
```

### Using docker-compose
```bash
# Start (from deployment directory)
cd deployment && docker-compose up -d

# View logs
cd deployment && docker-compose logs -f

# Stop
cd deployment && docker-compose down
```

## Environment Variables

The server uses environment variables from your `.env` file. Key variables include:

- `MODEL_NAME`: The HuggingFace model to use
- `AVAILABLE_VOICES`: Comma-separated list of available voices
- `TRT_*`: TensorRT-LLM configuration parameters
- `HF_TOKEN`: (Optional) HuggingFace token for private models

## Volume Mounts

The Docker setup includes several volume mounts:

- `/workspace/hf`: Model cache (persisted as named volume)
- `/app/logs`: Application logs
- `/app/outputs`: Output files

## API Endpoints

### TTS Generation
- `POST /v1/audio/speech/stream` - Generate speech (streaming PCM audio)
- `GET /v1/voices` - List available voices
- `WebSocket /v1/audio/speech/stream/ws` - WebSocket streaming
- `WebSocket /v1/tts` - WebSocket with word timestamps

### Performance Testing
- `POST /v1/benchmark` - Run performance benchmark
  - Parameters:
    - `text`: Text to benchmark
    - `voice`: Voice to use
    - `num_runs`: Number of test runs (1-20)
    - `warmup`: Whether to perform warmup run
  - Returns metrics including:
    - TTFB (Time to First Byte) in milliseconds
    - Total generation time
    - RTF (Real-Time Factor)
    - Audio duration and characteristics

### Health
- `GET /health` - Health check endpoint

## Performance Notes

- First startup will download the model (~50GB), subsequent runs use cached model
- The server includes health checks and will restart automatically if it fails
- Optimized for A100 GPUs with TensorRT-LLM for low latency (<160ms TTFB)
- Use the benchmark endpoint to measure actual performance on your hardware

## Troubleshooting

1. **GPU not detected**: Ensure NVIDIA Container Toolkit is installed
2. **Out of memory**: Adjust `TRT_FREE_GPU_MEMORY_FRACTION` in `.env`
3. **Model download fails**: Check `HF_TOKEN` if using private models
4. **Port already in use**: Change the port mapping in docker-compose.yml or docker-run.sh
