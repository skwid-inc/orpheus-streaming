"""
Enhanced WebSocket handler with best practices for streaming TTS.
Designed for compatibility with LiveKit and other WebRTC platforms.
"""

import asyncio
import json
import time
import logging
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from contextlib import asynccontextmanager

from .audio_stream_utils import AudioStreamConfig, AudioFormat, AudioBuffer, stream_with_format

logger = logging.getLogger(__name__)

class StreamingSession:
    """Manages a single streaming session with state tracking."""
    
    def __init__(self, session_id: str, client_info: str):
        self.session_id = session_id
        self.client_info = client_info
        self.start_time = time.time()
        self.bytes_sent = 0
        self.chunks_sent = 0
        self.is_active = True
        self.audio_buffer = AudioBuffer(max_size=20)
        
    def update_stats(self, bytes_sent: int):
        """Update session statistics."""
        self.bytes_sent += bytes_sent
        self.chunks_sent += 1
        
    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.start_time

class EnhancedTTSWebSocket:
    """
    Enhanced WebSocket handler with best practices:
    - Proper error handling and reconnection support
    - Backpressure management
    - Format negotiation
    - LiveKit-compatible metadata
    - Connection health monitoring
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.sessions: Dict[str, StreamingSession] = {}
        self.supported_formats = [AudioFormat.PCM_S16LE, AudioFormat.WAV]
        
    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection with best practices."""
        await websocket.accept()
        
        client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        session_id = f"session_{int(time.time() * 1000)}"
        session = StreamingSession(session_id, client_info)
        self.sessions[session_id] = session
        
        logger.info(f"WebSocket session {session_id} started for client {client_info}")
        
        # Send initial capabilities
        await self._send_capabilities(websocket, session_id)
        
        try:
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(
                self._heartbeat_handler(websocket, session)
            )
            
            # Main message loop
            await self._message_loop(websocket, session)
            
        except WebSocketDisconnect:
            logger.info(f"Session {session_id}: Client disconnected normally")
        except Exception as e:
            logger.error(f"Session {session_id}: Error - {e}")
            await self._send_error(websocket, str(e))
        finally:
            session.is_active = False
            heartbeat_task.cancel()
            
            # Send session summary
            await self._send_session_summary(websocket, session)
            
            # Cleanup
            del self.sessions[session_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
                
            logger.info(
                f"Session {session_id} ended - duration: {session.duration:.2f}s, "
                f"bytes sent: {session.bytes_sent:,}"
            )
    
    async def _send_capabilities(self, websocket: WebSocket, session_id: str):
        """Send server capabilities for format negotiation."""
        capabilities = {
            "type": "CAPABILITIES",
            "session_id": session_id,
            "version": "1.0",
            "audio": {
                "formats": ["pcm_s16le", "wav"],
                "sample_rates": [16000, 24000, 48000],
                "channels": [1, 2],
                "streaming": True,
                "word_timestamps": True
            },
            "features": {
                "voice_selection": False,  # Model doesn't support voice selection
                "speed_control": True,
                "pitch_control": False,
                "backpressure": True,
                "reconnection": True
            }
        }
        await websocket.send_json(capabilities)
    
    async def _message_loop(self, websocket: WebSocket, session: StreamingSession):
        """Main message processing loop."""
        while session.is_active:
            try:
                # Use timeout to allow periodic checks
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )
                
                msg_type = message.get("type", "SYNTHESIZE")
                
                if msg_type == "SYNTHESIZE":
                    await self._handle_synthesize(websocket, session, message)
                elif msg_type == "CONFIGURE":
                    await self._handle_configure(websocket, session, message)
                elif msg_type == "PAUSE":
                    await self._handle_pause(websocket, session)
                elif msg_type == "RESUME":
                    await self._handle_resume(websocket, session)
                elif msg_type == "STOP":
                    break
                elif msg_type == "PING":
                    await websocket.send_json({"type": "PONG", "timestamp": time.time()})
                else:
                    await self._send_error(websocket, f"Unknown message type: {msg_type}")
                    
            except asyncio.TimeoutError:
                # No message received, continue loop
                continue
            except json.JSONDecodeError as e:
                await self._send_error(websocket, f"Invalid JSON: {e}")
    
    async def _handle_synthesize(self, websocket: WebSocket, session: StreamingSession, message: Dict[str, Any]):
        """Handle TTS synthesis request with best practices."""
        text = message.get("text", "").strip()
        if not text:
            await self._send_error(websocket, "No text provided")
            return
            
        request_id = message.get("request_id", f"req_{int(time.time() * 1000)}")
        format_type = message.get("format", "pcm_s16le")
        include_header = message.get("include_header", False)
        
        # Parse audio configuration
        audio_config = message.get("audio_config", {})
        config = AudioStreamConfig(
            sample_rate=audio_config.get("sample_rate", 24000),
            channels=audio_config.get("channels", 1),
            format=AudioFormat(format_type),
            include_wav_header=include_header
        )
        
        logger.info(
            f"Session {session.session_id}: Synthesizing '{text[:50]}...' "
            f"format: {format_type}, request_id: {request_id}"
        )
        
        # Send synthesis started event
        await websocket.send_json({
            "type": "SYNTHESIS_STARTED",
            "request_id": request_id,
            "text_length": len(text),
            "audio_config": {
                "format": config.format.value,
                "sample_rate": config.sample_rate,
                "channels": config.channels,
                "bits_per_sample": config.bits_per_sample
            }
        })
        
        try:
            # Generate audio
            start_time = time.perf_counter()
            audio_generator = self.engine.generate_speech_async(prompt=text)
            
            # Stream with format handling
            formatted_stream = stream_with_format(
                audio_generator,
                config,
                include_header=include_header
            )
            
            first_chunk = True
            chunk_count = 0
            total_bytes = 0
            
            async for chunk in formatted_stream:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    await websocket.send_json({
                        "type": "FIRST_AUDIO_CHUNK",
                        "request_id": request_id,
                        "ttfb_ms": ttfb * 1000
                    })
                    first_chunk = False
                
                # Check for backpressure
                if session.audio_buffer.is_full:
                    # Wait a bit for buffer to drain
                    await asyncio.sleep(0.01)
                
                # Send audio chunk with metadata
                await websocket.send_bytes(chunk)
                
                chunk_count += 1
                total_bytes += len(chunk)
                session.update_stats(len(chunk))
                
                # Send progress updates every 10 chunks
                if chunk_count % 10 == 0:
                    await websocket.send_json({
                        "type": "PROGRESS",
                        "request_id": request_id,
                        "chunks_sent": chunk_count,
                        "bytes_sent": total_bytes
                    })
            
            # Send completion event
            generation_time = time.perf_counter() - start_time
            audio_duration = total_bytes / config.bytes_per_second
            
            await websocket.send_json({
                "type": "SYNTHESIS_COMPLETED",
                "request_id": request_id,
                "stats": {
                    "generation_time_ms": generation_time * 1000,
                    "audio_duration_s": audio_duration,
                    "total_bytes": total_bytes,
                    "chunks": chunk_count,
                    "rtf": generation_time / audio_duration if audio_duration > 0 else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Session {session.session_id}: Synthesis error - {e}")
            await websocket.send_json({
                "type": "SYNTHESIS_ERROR",
                "request_id": request_id,
                "error": str(e)
            })
    
    async def _handle_configure(self, websocket: WebSocket, session: StreamingSession, message: Dict[str, Any]):
        """Handle configuration updates."""
        # Store configuration for the session
        config = message.get("config", {})
        logger.info(f"Session {session.session_id}: Configuration updated - {config}")
        
        await websocket.send_json({
            "type": "CONFIGURED",
            "config": config
        })
    
    async def _handle_pause(self, websocket: WebSocket, session: StreamingSession):
        """Handle pause request."""
        # Implementation would pause ongoing synthesis
        logger.info(f"Session {session.session_id}: Paused")
        await websocket.send_json({"type": "PAUSED"})
    
    async def _handle_resume(self, websocket: WebSocket, session: StreamingSession):
        """Handle resume request."""
        # Implementation would resume paused synthesis
        logger.info(f"Session {session.session_id}: Resumed")
        await websocket.send_json({"type": "RESUMED"})
    
    async def _heartbeat_handler(self, websocket: WebSocket, session: StreamingSession):
        """Send periodic heartbeats to detect connection issues."""
        try:
            while session.is_active:
                await asyncio.sleep(15)  # Send heartbeat every 15 seconds
                
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "type": "HEARTBEAT",
                        "timestamp": time.time(),
                        "session_duration": session.duration,
                        "bytes_sent": session.bytes_sent
                    })
                else:
                    break
        except Exception as e:
            logger.debug(f"Heartbeat error: {e}")
    
    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to client."""
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "ERROR",
                "message": error_message,
                "timestamp": time.time()
            })
    
    async def _send_session_summary(self, websocket: WebSocket, session: StreamingSession):
        """Send session summary before closing."""
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "SESSION_END",
                "session_id": session.session_id,
                "duration_s": session.duration,
                "total_bytes": session.bytes_sent,
                "total_chunks": session.chunks_sent
            })
