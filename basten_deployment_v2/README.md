# Orpheus TTS - Baseten Deployment

This directory contains the Baseten/Truss deployment configuration for the Orpheus TTS model with TensorRT-LLM optimization.

## Overview

This deployment packages the Orpheus 3B TTS model for deployment on Baseten, providing:
- Low-latency streaming TTS generation
- TensorRT-LLM optimization for fast inference
- Support for streaming audio output
- PCM audio format at 24kHz sample rate

## Prerequisites

1. Install Truss CLI:
```bash
pip install --upgrade truss 'pydantic>=2.0.0'
```

2. Set up your Baseten API key:
```bash
export BASETEN_API_KEY="your-api-key-here"
```

## Deployment Steps

1. **Navigate to the deployment directory:**
```bash
cd baseten-deployment
```

2. **Deploy to Baseten:**
```bash
truss push
```

3. **Test the deployment:**
Once deployed, you'll receive a WebSocket URL for streaming TTS with word timestamps.

### WebSocket Usage (Recommended)

The model uses WebSocket for real-time streaming with word timestamps:

```python
import asyncio
import websockets
import json
import base64

async def test_tts_websocket():
    # Your Baseten WebSocket URL
    ws_url = "wss://model-{model-id}.api.baseten.co/development/handle_websocket"
    headers = {"Authorization": f"Api-Key {api_key}"}
    
    async with websockets.connect(ws_url, extra_headers=headers) as websocket:
        # Send text for TTS generation
        await websocket.send(json.dumps({
            "input": "Hello world, this is a test.",
            "continue": True,
            "segment_id": "segment_1"
        }))
        
        # Receive streaming responses
        while True:
            message = await websocket.recv()
            
            if isinstance(message, bytes):
                # Audio chunk received
                with open("output.pcm", "ab") as f:
                    f.write(message)
            else:
                # JSON message (start/end/timestamps)
                data = json.loads(message)
                print(f"Message: {data}")
                
                if data.get("done"):
                    break
        
        # End the stream
        await websocket.send(json.dumps({"continue": False}))

# Run the WebSocket client
asyncio.run(test_tts_websocket())
```

### HTTP Usage (Non-streaming only)

For non-streaming requests:

```bash
curl -X POST https://model-{model-id}.api.baseten.co/development/predict \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world, this is a test.",
    "stream": false
  }'
```

## API Usage

### WebSocket Protocol

The WebSocket endpoint follows this message flow:

**Client → Server:**
```json
{
  "input": "Text to convert to speech",
  "continue": true,    // Set to false to end the connection
  "segment_id": "unique_id"  // Optional segment identifier
}
```

**Server → Client Messages:**

1. **Start message:**
```json
{
  "type": "start",
  "segment_id": "unique_id"
}
```

2. **Audio chunks:** Binary PCM audio data (24kHz, 16-bit, mono)

3. **Word timestamps (if using handle_websocket):**
```json
{
  "type": "timestamps",
  "segment_id": "unique_id",
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.5, "end": 1.0}
  ]
}
```

4. **End message:**
```json
{
  "type": "end",
  "segment_id": "unique_id"
}
```

5. **Final message (when continue=false):**
```json
{
  "done": true
}
```

### HTTP Endpoint (Non-streaming only)

For simple non-streaming requests:

```json
{
  "input": "Text to convert to speech",
  "stream": false
}
```

Response:
```json
{
  "audio": "base64_encoded_pcm_data",
  "content_type": "audio/pcm",
  "sample_rate": 24000
}
```

### Example Usage

```python
import requests
import base64

# Your Baseten model URL and API key
model_url = "https://model-{model-id}.api.baseten.co/development/predict"
api_key = "your-api-key"

# Generate speech
response = requests.post(
    model_url,
    headers={"Authorization": f"Api-Key {api_key}"},
    json={
        "input": "Welcome to Orpheus text to speech system.",
        "stream": False
    }
)

# Save audio to file
if response.status_code == 200:
    result = response.json()
    audio_data = base64.b64decode(result["audio"])
    with open("output.pcm", "wb") as f:
        f.write(audio_data)
```

## Configuration

The deployment is configured via `config.yaml`:

- **Model**: `canopylabs/orpheus-3b-0.1-ft`
- **GPU**: A100 (can be adjusted in config)
- **Memory**: 16GB
- **Python**: 3.10
- **TensorRT-LLM**: Optimized for low latency

### Environment Variables

Key configuration parameters can be adjusted via environment variables:

- `TRT_TEMPERATURE`: Generation temperature (default: 0.1)
- `TRT_MAX_TOKENS`: Maximum tokens to generate (default: 1200)
- `TRT_DTYPE`: Model precision (default: bfloat16)
- `TRT_MAX_BATCH_SIZE`: Maximum batch size (default: 4)

## Audio Format

The model outputs:
- **Format**: PCM (raw audio)
- **Sample Rate**: 24,000 Hz
- **Bit Depth**: 16-bit
- **Channels**: Mono

To play the audio, you can use:
```bash
# Using ffplay (requires ffmpeg)
ffplay -f s16le -ar 24000 -ac 1 output.pcm

# Convert to WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

## Performance

Expected performance on H100:
- **TTFB**: <160ms (Time to First Byte) with TensorRT-LLM
- **RTF**: >1.0x real-time (Real-Time Factor)
- **Throughput**: Optimized for single-stream low latency with WebSocket streaming

## Troubleshooting

1. **CUDA/TensorRT Issues**: Ensure the deployment has access to GPU resources
2. **Memory Issues**: Adjust `TRT_FREE_GPU_MEMORY_FRACTION` if needed
3. **Token Generation**: Check `TRT_STOP_TOKEN_IDS` if generation doesn't stop properly

## Development

To make changes:

1. Modify the code in `model/model.py` or update configurations
2. Test locally with: `truss predict -d . --input '{"input": "test"}'`
3. Push updates: `truss push`

## Support

For issues specific to:
- **Baseten deployment**: Check [Baseten docs](https://docs.baseten.co)
- **Model issues**: Refer to the main Orpheus repository
- **TensorRT-LLM**: See NVIDIA TensorRT-LLM documentation
