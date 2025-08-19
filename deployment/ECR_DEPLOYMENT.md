# ECR Deployment Guide

## ECR Repository Details

- **Repository URI**: `308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts`
- **Region**: `us-west-2`
- **Image Tag**: `latest`

## Pulling the Image from ECR

### 1. Authenticate with ECR
```bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 308804103509.dkr.ecr.us-west-2.amazonaws.com
```

### 2. Pull the Image
```bash
docker pull 308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:latest
```

### 3. Run the Container
```bash
docker run --gpus all \
  --name orpheus-tts \
  --rm \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HUB_OFFLINE=0 \
  -p 9090:9090 \
  -v orpheus_hf_cache:/workspace/hf \
  308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:latest
```

## Using with Docker Compose

Update your `docker-compose.yml` to use the ECR image:

```yaml
version: '3.8'

services:
  orpheus-tts:
    image: 308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:latest
    container_name: orpheus-tts-server
    runtime: nvidia
    env_file:
      - ../.env
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - TRANSFORMERS_OFFLINE=0
      - HF_HUB_OFFLINE=0
    ports:
      - "9090:9090"
    volumes:
      - orpheus_hf_cache:/workspace/hf
      - ../logs:/app/logs
      - ../outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  orpheus_hf_cache:
    driver: local
```

## Deployment on AWS ECS/Fargate

For deploying on AWS ECS with GPU support, ensure:
1. Use GPU-enabled task definitions
2. Set proper IAM roles for ECR access
3. Configure the task with sufficient memory (at least 32GB recommended)
4. Use P3 or G4 instance types for GPU support

## Notes

- The image is ~25GB compressed, so initial pulls may take time
- Model weights are downloaded on first run and cached in the volume
- Ensure your EC2 instances or ECS tasks have proper IAM roles for ECR access
