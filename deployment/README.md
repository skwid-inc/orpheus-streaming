# Deployment Files

This directory contains all files needed to deploy the Orpheus TTS Server using Docker.

## Files

- `Dockerfile` - Docker image definition optimized for A100 GPUs
- `docker-compose.yml` - Docker Compose configuration for easy deployment
- `.dockerignore` - Files to exclude from Docker build context
- `docker-build.sh` - Script to build the Docker image
- `docker-run.sh` - Script to run the container with proper settings
- `test_docker_deployment.sh` - Script to test the deployment
- `DOCKER_DEPLOYMENT.md` - Complete deployment documentation

## Quick Start

From the project root directory:

```bash
# Build the image
./deployment/docker-build.sh

# Run the server
./deployment/docker-run.sh

# Or use docker-compose
cd deployment && docker-compose up -d
```

See [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) for detailed instructions.
