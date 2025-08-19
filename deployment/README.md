# Deployment Files

This directory contains all files needed to deploy the Orpheus TTS Server using Docker.

## Files

- `Dockerfile` - Docker image definition optimized for GPU inference
- `docker-compose.yml` - Docker Compose configuration for easy deployment
- `.dockerignore` - Files to exclude from Docker build context
- `docker-build.sh` - Script to build the Docker image
- `docker-run.sh` - Script to run the container with proper settings
- `test_docker_deployment.sh` - Script to test the deployment
- `DOCKER_DEPLOYMENT.md` - Complete deployment documentation
- `ECR_DEPLOYMENT.md` - Guide for using the pre-built ECR image

## Quick Start - Choose Your Path

### Option 1: Use Pre-built Image (Recommended)

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 308804103509.dkr.ecr.us-west-2.amazonaws.com

# Pull and run
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

### Option 2: Build from Source

From the project root directory:

```bash
# Build the image
./deployment/docker-build.sh

# Run the server
./deployment/docker-run.sh

# Or use docker-compose
cd deployment && docker compose up -d
```

See [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) for detailed instructions.
