#!/bin/bash
# orpheus_multi_gpu.sh - Main management script for Orpheus multi-GPU setup

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to show usage
usage() {
    echo -e "${BLUE}Orpheus Multi-GPU Manager${NC}"
    echo ""
    echo "Usage: $0 {start|stop|restart|status|cache|install-nginx|test}"
    echo ""
    echo "Commands:"
    echo "  start         - Start all Orpheus instances"
    echo "  stop          - Stop all Orpheus instances"
    echo "  restart       - Restart all Orpheus instances"
    echo "  status        - Show status of all instances"
    echo "  cache         - Check model/engine cache status"
    echo "  install-nginx - Install and configure nginx (run once)"
    echo "  test          - Run a simple test request"
    echo ""
    exit 1
}

# Install nginx configuration
install_nginx() {
    echo -e "${BLUE}Installing nginx configuration...${NC}"
    
    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        echo -e "${RED}Nginx is not installed. Install it with:${NC}"
        echo "sudo apt-get update && sudo apt-get install -y nginx"
        exit 1
    fi
    
    # Copy configuration
    sudo cp "$SCRIPT_DIR/nginx/orpheus_lb.conf" /etc/nginx/sites-available/
    sudo ln -sf /etc/nginx/sites-available/orpheus_lb.conf /etc/nginx/sites-enabled/
    
    # Test configuration
    if sudo nginx -t; then
        echo -e "${GREEN}Nginx configuration is valid${NC}"
        sudo systemctl reload nginx
        echo -e "${GREEN}Nginx reloaded successfully${NC}"
    else
        echo -e "${RED}Nginx configuration error${NC}"
        exit 1
    fi
}

# Start all services
start_services() {
    echo -e "${BLUE}Starting Orpheus multi-GPU setup...${NC}"
    
    # Start Orpheus instances
    "$SCRIPT_DIR/launch_multi_gpu.sh"
    
    # Start nginx if not running
    if ! pgrep nginx > /dev/null; then
        echo -e "${YELLOW}Starting nginx...${NC}"
        sudo systemctl start nginx
    fi
    
    echo -e "\n${GREEN}All services started!${NC}"
    echo -e "Load balancer is available at: ${BLUE}http://localhost:8080${NC}"
}

# Stop all services
stop_services() {
    echo -e "${BLUE}Stopping Orpheus multi-GPU setup...${NC}"
    
    # Stop Orpheus instances
    "$SCRIPT_DIR/stop_multi_gpu.sh"
    
    echo -e "${GREEN}All Orpheus instances stopped${NC}"
}

# Test the setup
test_setup() {
    echo -e "${BLUE}Testing Orpheus multi-GPU setup...${NC}"
    
    # Check if load balancer is responding
    if curl -s -o /dev/null -w '' "http://localhost:8080/docs" 2>/dev/null; then
        echo -e "${GREEN}✓ Load balancer is responding${NC}"
    else
        echo -e "${RED}✗ Load balancer is not responding${NC}"
        exit 1
    fi
    
    # Test TTS endpoint
    echo -e "\n${BLUE}Sending test TTS request...${NC}"
    response=$(curl -s -X POST "http://localhost:8080/v1/audio/speech/stream" \
        -H "Content-Type: application/json" \
        -d '{"input": "Hello from multi GPU setup!"}' \
        -o /tmp/test_audio.pcm \
        -w "%{http_code}")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ TTS request successful${NC}"
        echo "Audio saved to: /tmp/test_audio.pcm"
        echo "File size: $(stat -c%s /tmp/test_audio.pcm 2>/dev/null || echo 'N/A') bytes"
    else
        echo -e "${RED}✗ TTS request failed (HTTP $response)${NC}"
    fi
}

# Make scripts executable
SCRIPT_DIR="$(dirname "$0")"
chmod +x "$SCRIPT_DIR"/launch_multi_gpu.sh "$SCRIPT_DIR"/stop_multi_gpu.sh "$SCRIPT_DIR"/check_cache.sh

# Main command handling
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_services
        ;;
    status)
        echo "Status script was removed to keep things minimal"
    echo "Check running instances with: ps aux | grep uvicorn"
        ;;
    cache)
        "$SCRIPT_DIR/check_cache.sh"
        ;;
    install-nginx)
        install_nginx
        ;;
    test)
        test_setup
        ;;
    *)
        usage
        ;;
esac
