#!/bin/bash
set -e

echo "=== Testing Orpheus TTS 8xGPU Deployment ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cd "$(dirname "$0")"

# Function to wait for containers to be healthy
wait_for_healthy() {
    local service=$1
    local max_attempts=30
    local attempt=0
    
    echo -n "Waiting for $service to be healthy"
    while [ $attempt -lt $max_attempts ]; do
        if docker compose ps | grep -q "$service.*healthy"; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 5
        ((attempt++))
    done
    echo -e " ${RED}✗${NC}"
    return 1
}

# Test docker build workflow
echo -e "${YELLOW}Testing Docker Build Workflow${NC}"
echo "================================"

echo "1. Starting containers with build..."
./start-8x.sh

echo ""
echo "2. Waiting for all services to be healthy..."
all_healthy=true
for i in {0..7}; do
    if ! wait_for_healthy "orpheus-tts-$i"; then
        all_healthy=false
        echo -e "${RED}Service orpheus-tts-$i failed to become healthy${NC}"
    fi
done

if ! wait_for_healthy "nginx"; then
    all_healthy=false
    echo -e "${RED}NGINX failed to become healthy${NC}"
fi

if [ "$all_healthy" = true ]; then
    echo -e "${GREEN}All services are healthy!${NC}"
    
    echo ""
    echo "3. Testing endpoints..."
    
    # Test NGINX health endpoint
    echo -n "Testing NGINX health endpoint... "
    if curl -s http://localhost:18080/health | grep -q "healthy"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
    
    # Test main API endpoint through NGINX
    echo -n "Testing main API endpoint... "
    if curl -s http://localhost:18080/docs > /dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
    
    echo ""
    echo "4. Container status:"
    docker compose ps
else
    echo -e "${RED}Some services failed to start properly${NC}"
    echo ""
    echo "Container status:"
    docker compose ps
    echo ""
    echo "Recent logs:"
    docker compose logs --tail=50
fi

echo ""
echo "5. Stopping containers..."
./stop-8x.sh

echo ""
echo -e "${GREEN}Docker build workflow test completed!${NC}"
echo ""

# Test docker pull workflow
echo -e "${YELLOW}Testing Docker Pull Workflow${NC}"
echo "================================"

echo "Note: This test requires the image to be available in the ECR registry."
echo "Current image: 308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:latest"
echo ""

read -p "Do you want to test the pull workflow? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "1. Starting containers with pull..."
    ./start-8x-pull.sh
    
    echo ""
    echo "2. Waiting for services to be healthy..."
    sleep 10
    
    echo "3. Container status:"
    docker compose -f docker-compose.pull.yml ps
    
    echo ""
    echo "4. Stopping containers..."
    ./stop-8x.sh
    
    echo -e "${GREEN}Docker pull workflow test completed!${NC}"
else
    echo "Skipping pull workflow test."
fi

echo ""
echo "=== All tests completed ==="
