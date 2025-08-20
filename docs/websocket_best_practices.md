# WebSocket TTS Streaming Best Practices & LiveKit Compatibility

## Current Implementation Analysis

### ✅ What We Have
1. **Low-latency streaming** - Audio chunks are streamed as generated
2. **Async processing** - Non-blocking I/O for better concurrency
3. **Basic error handling** - Connection errors and disconnects handled
4. **Logging** - Comprehensive request/response logging
5. **Word timestamps** - Synchronized word-level timing information

### ❌ What's Missing for Production/LiveKit

#### 1. **Audio Format Flexibility**
- **Current**: Raw PCM only (24kHz, 16-bit, mono)
- **Needed**: Multiple format support (Opus for WebRTC, WAV with headers)
- **Why**: LiveKit uses WebRTC which prefers Opus codec

#### 2. **Connection Resilience**
- **Current**: Basic connection handling
- **Missing**:
  - Automatic reconnection with session resumption
  - Heartbeat/keepalive mechanism
  - Connection quality monitoring
  - Graceful degradation under poor network

#### 3. **Backpressure Management**
- **Current**: Direct streaming without buffering
- **Needed**: Buffer management to handle slow clients
- **Why**: Prevents memory issues and dropped audio

#### 4. **Protocol Enhancement**
- **Current**: Simple request/response
- **Needed**:
  - Message type system (SYNTHESIZE, PAUSE, RESUME, CONFIGURE)
  - Request IDs for tracking multiple concurrent requests
  - Progress updates during long syntheses
  - Capability negotiation

#### 5. **WebRTC Integration**
- **Current**: WebSocket-only
- **Needed**: WebRTC data channels for lower latency
- **Why**: LiveKit is WebRTC-based, offers better real-time performance

## Implementation Recommendations

### 1. **Immediate Improvements** (WebSocket-based)

```python
# Add to main.py
from src.websocket_best_practices import EnhancedTTSWebSocket

# Enhanced endpoint with best practices
enhanced_tts_handler = EnhancedTTSWebSocket(engine)

@app.websocket("/v1/tts/enhanced")
async def enhanced_tts_websocket(websocket: WebSocket):
    """Enhanced WebSocket endpoint following best practices."""
    await enhanced_tts_handler.handle_websocket(websocket)
```

### 2. **Audio Format Support**

```python
# Client can request different formats
{
    "type": "SYNTHESIZE",
    "text": "Hello world",
    "format": "opus",  // or "pcm_s16le", "wav"
    "audio_config": {
        "sample_rate": 48000,  // LiveKit typically uses 48kHz
        "channels": 1
    }
}
```

### 3. **LiveKit Integration Path**

For full LiveKit compatibility, consider:

1. **Use LiveKit Agents SDK**:
```python
from livekit import agents

class OrpheusTTSAgent(agents.Plugin):
    def __init__(self, engine):
        self.engine = engine
        
    async def process(self, text: str) -> agents.AudioFrame:
        # Convert TTS output to LiveKit audio frames
        pass
```

2. **Implement STT/TTS Plugin Interface**:
- LiveKit expects specific interfaces for real-time agents
- Audio should be in Opus format at 48kHz
- Support for interruption and cancellation

3. **WebRTC Data Channels**:
- Lower latency than WebSocket
- Built-in congestion control
- Better for real-time audio

### 4. **Session Management**

```python
# Support for stateful sessions
{
    "type": "CONFIGURE",
    "config": {
        "speaking_rate": 1.2,
        "pitch": 0,
        "session_id": "unique-session-id"
    }
}
```

### 5. **Error Recovery**

```python
# Automatic retry with exponential backoff
async def with_retry(func, max_retries=3):
    for i in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if i == max_retries - 1:
                raise
            await asyncio.sleep(2 ** i)
```

## Testing Enhanced Implementation

```python
# Test client for enhanced WebSocket
import asyncio
import websockets
import json

async def test_enhanced_websocket():
    uri = "ws://localhost:9090/v1/tts/enhanced"
    
    async with websockets.connect(uri) as ws:
        # Receive capabilities
        caps = await ws.recv()
        print(f"Server capabilities: {json.loads(caps)}")
        
        # Send synthesis request
        await ws.send(json.dumps({
            "type": "SYNTHESIZE",
            "text": "Testing enhanced WebSocket",
            "format": "pcm_s16le",
            "request_id": "test-123"
        }))
        
        # Handle responses
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                print(f"Received audio chunk: {len(msg)} bytes")
            else:
                data = json.loads(msg)
                print(f"Message: {data['type']}")
                if data['type'] == 'SYNTHESIS_COMPLETED':
                    break

asyncio.run(test_enhanced_websocket())
```

## Performance Considerations

1. **Chunking Strategy**:
   - Current: 4KB chunks
   - Optimal: 20-50ms of audio per chunk for real-time
   - At 24kHz: ~1920-4800 bytes per chunk

2. **Latency Targets**:
   - TTFB: < 200ms for real-time feel
   - End-to-end: < 500ms for conversational AI

3. **Concurrent Connections**:
   - Implement connection pooling
   - Rate limiting per client
   - Resource monitoring

## Security Considerations

1. **Authentication**: Add JWT or API key validation
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: Sanitize text input
4. **SSL/TLS**: Use WSS in production

## Monitoring & Metrics

Track these metrics for production:
- Connection count and duration
- TTFB percentiles (p50, p95, p99)
- Audio generation RTF (Real-Time Factor)
- Error rates by type
- Network quality indicators

## Conclusion

While the current implementation works well for basic streaming, production use with LiveKit requires:
1. Multiple audio format support (especially Opus)
2. Robust error handling and reconnection
3. WebRTC integration for lowest latency
4. Session state management
5. Comprehensive monitoring

The enhanced WebSocket handler provides many of these features while maintaining compatibility with the existing architecture.
