# Deployment Structure

This deployment setup is organized to keep all Docker-related files in a dedicated directory while maintaining proper build contexts.

## Directory Layout

```
orpheus-streaming/           # Project root
├── .env                     # Environment variables (used by Docker)
├── src/                     # Source code
│   ├── benchmark_api.py     # NEW: Benchmark API service
│   └── ...                  # Other source files
├── main.py                  # Main application
├── test_benchmark_api.py    # NEW: Benchmark API test script
└── deployment/              # All Docker deployment files
    ├── Dockerfile           # Docker image definition
    ├── docker-compose.yml   # Docker Compose configuration
    ├── .dockerignore        # Build exclusions
    ├── docker-build.sh      # Build script
    ├── docker-run.sh        # Run script
    ├── test_docker_deployment.sh  # Deployment test
    ├── DOCKER_DEPLOYMENT.md # Deployment documentation
    ├── ECR_DEPLOYMENT.md    # ECR pre-built image guide
    ├── QUICKSTART.md        # Quick start guide
    └── README.md            # Deployment overview
```

## Key Features Added

1. **Benchmark API Endpoint** (`/v1/benchmark`)
   - Performs multiple TTS generation runs
   - Returns performance metrics (TTFB, RTF, etc.)
   - Configurable number of runs and warmup

2. **Organized Deployment**
   - All Docker files in `deployment/` directory
   - Scripts handle path navigation automatically
   - docker-compose uses parent context for builds

3. **Two Deployment Paths**
   - **Pre-built Image**: Use the ECR image for quick deployment
   - **Build from Source**: Build your own image with customizations
   - Clear documentation for both approaches

## Usage

### Using Pre-built Image (Recommended)

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 308804103509.dkr.ecr.us-west-2.amazonaws.com

# Run directly
docker run --gpus all --env-file .env -p 9090:9090 308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:latest
```

### Building from Source

All commands should be run from the project root:

```bash
# Build
./deployment/docker-build.sh

# Run
./deployment/docker-run.sh

# Test benchmark API
python test_benchmark_api.py
```
