#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p ../logs/tts-{0,1,2,3,4,5,6,7} ../outputs/tts-{0,1,2,3,4,5,6,7}

docker compose -f docker-compose.pull.yml up -d

echo "Stack started with NGINX load balancer on :18080 (pull mode)"
echo "Individual instances running on internal ports 9090-9097"

