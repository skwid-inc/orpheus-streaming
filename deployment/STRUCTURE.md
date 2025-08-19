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
    └── DOCKER_DEPLOYMENT.md # Deployment documentation
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

## Usage

All commands should be run from the project root:

```bash
# Build
./deployment/docker-build.sh

# Run
./deployment/docker-run.sh

# Test benchmark API
python test_benchmark_api.py
```
