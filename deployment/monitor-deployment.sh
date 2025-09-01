#!/bin/bash

echo "=== Orpheus TTS 8xGPU Deployment Monitor ==="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

while true; do
    clear
    echo "=== Orpheus TTS 8xGPU Deployment Monitor ==="
    echo "Time: $(date)"
    echo ""
    
    # Test NGINX
    echo -n "NGINX Load Balancer: "
    if curl -s http://localhost:18080/nginx-health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Working${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
    fi
    
    # Test main endpoint
    echo -n "Main Endpoint: "
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:18080/)
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ Ready (HTTP $response)${NC}"
    elif [ "$response" = "502" ]; then
        echo -e "${YELLOW}⏳ Services loading (HTTP $response)${NC}"
    else
        echo -e "${RED}✗ Error (HTTP $response)${NC}"
    fi
    
    echo ""
    echo "Container Status:"
    docker compose ps --format "table {{.Service}}\t{{.Status}}"
    
    echo ""
    echo "Recent activity from TTS-0:"
    docker compose logs --tail=3 orpheus-tts-0 2>/dev/null | tail -3
    
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 10
done
