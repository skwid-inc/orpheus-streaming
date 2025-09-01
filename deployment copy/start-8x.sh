#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p ../logs/tts-{0,1,2,3,4,5,6,7} ../outputs/tts-{0,1,2,3,4,5,6,7}

docker compose up -d --build

echo "Stack started with NGINX load balancer on :18080"
echo "Individual instances running on internal ports 9090-9097"

