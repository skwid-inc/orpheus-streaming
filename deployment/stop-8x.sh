#!/bin/bash
set -e

cd "$(dirname "$0")"

docker compose down || true
docker compose -f docker-compose.pull.yml down || true

echo "Stack stopped"

