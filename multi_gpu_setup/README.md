# Multi-GPU Setup

Run Orpheus TTS across 8 GPUs with nginx load balancing.

## Quick Start

```bash
# From the parent directory:
./multi_gpu_setup/orpheus_multi_gpu.sh start

# Or from this directory:
./orpheus_multi_gpu.sh start
```

## Commands

- `start` - Start all 8 instances
- `stop` - Stop all instances  
- `restart` - Restart all instances
- `cache` - Check model/engine cache
- `install-nginx` - Configure nginx (one time)
- `test` - Test the setup

## Configuration

Edit `multi_gpu.env` to adjust settings.

## Access

Load balancer: http://localhost:8080
