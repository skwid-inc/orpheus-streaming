#!/bin/bash

echo "Testing Orpheus TTS Docker Deployment..."
echo "========================================"

# Check if server is running
echo -n "1. Checking health endpoint... "
if curl -s -f http://localhost:9090/health > /dev/null; then
    echo "✓ Server is healthy"
else
    echo "✗ Server is not responding"
    exit 1
fi

# Get available voices
echo -n "2. Fetching available voices... "
voices=$(curl -s http://localhost:9090/v1/voices)
if [ $? -eq 0 ]; then
    echo "✓ Success"
    echo "   Available voices: $(echo $voices | jq -r '.voices[].name' | tr '\n' ', ' | sed 's/,$//')"
else
    echo "✗ Failed to fetch voices"
fi

# Test TTS generation
echo -n "3. Testing TTS generation... "
curl -s -X POST http://localhost:9090/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a test of the Orpheus text to speech system running in Docker.",
    "voice": "tara"
  }' \
  --output test_output.pcm

if [ -f test_output.pcm ] && [ -s test_output.pcm ]; then
    echo "✓ Audio generated successfully"
    echo "   Output saved to: test_output.pcm ($(stat -c%s test_output.pcm) bytes)"
    rm test_output.pcm
else
    echo "✗ Failed to generate audio"
fi

echo ""
echo "Deployment test complete!"
