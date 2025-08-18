"""
FastAPI WebSocket endpoint with word timestamps for Orpheus.
"""

import asyncio
import json
import time
import logging
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from typing import Optional, AsyncGenerator

from .word_timestamps import WordTimestampGenerator

logger = logging.getLogger(__name__)

# Audio constants
SR = 24000
BYTES_PER_SAMPLE = 2
CHANNELS = 1
BYTES_PER_SEC = SR * BYTES_PER_SAMPLE * CHANNELS


class TTSWithTimestamps:
    """TTS handler with word timestamp generation."""
    
    def __init__(self, engine):
        self.engine = engine
        self.timestamp_gen = WordTimestampGenerator()
    
    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection with word timestamps."""
        await websocket.accept()
        logger.info("WebSocket connection opened with timestamps support")
        
        try:
            while True:
                # Receive request
                data = await websocket.receive_json()
                
                if not data.get("continue", True):
                    logger.info("End of stream requested")
                    break
                
                text = data.get("text", data.get("input", "")).strip()
                if not text:
                    logger.info("Empty text received, skipping")
                    continue
                
                voice_id = data.get("voice_id", data.get("voice", "tara"))
                speaking_rate = data.get("speaking_rate", "normal")
                segment_id = data.get("segment_id", "default")
                
                # Process the request
                await self._process_tts_request(
                    websocket, text, voice_id, speaking_rate, segment_id
                )
                
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "ERROR", "message": str(e)})
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
    
    async def _process_tts_request(self, websocket: WebSocket, text: str, 
                                  voice_id: str, speaking_rate: str, segment_id: str):
        """Process a single TTS request with timestamps."""
        
        # Send metadata
        await websocket.send_json({
            "type": "META",
            "sample_rate": SR,
            "channels": CHANNELS,
            "format": "pcm_s16le",
            "segment_id": segment_id
        })
        
        # Create timeline
        timeline = self.timestamp_gen.create_timeline(text, voice_id, speaking_rate)
        
        if not timeline:
            logger.warning("Failed to create timeline")
            await websocket.send_json({"type": "ERROR", "message": "Failed to create timeline"})
            return
        
        # Send initial timeline
        await websocket.send_json(timeline.get_timeline_event())
        
        # Start synthesis
        start_time = time.perf_counter()
        bytes_sent = 0
        last_rescale_time = 0.0
        rescale_interval = timeline.voice_profile.rescale_interval_ms / 1000.0
        
        try:
            # Generate audio
            audio_generator = self.engine.generate_speech_async(
                prompt=text,
                voice=voice_id
            )
            
            first_chunk = True
            async for audio_chunk in audio_generator:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    logger.info(f"TTFB: {ttfb*1000:.2f}ms")
                    first_chunk = False
                
                # Send audio chunk
                await websocket.send_bytes(audio_chunk)
                bytes_sent += len(audio_chunk)
                
                # Calculate audio time
                audio_seconds = bytes_sent / BYTES_PER_SEC
                
                # Rescale timeline periodically
                if audio_seconds - last_rescale_time >= rescale_interval:
                    if timeline.rescale_to_audio_time(audio_seconds):
                        await websocket.send_json(timeline.get_timeline_event())
                    last_rescale_time = audio_seconds
            
            # Final timeline update
            final_audio_seconds = bytes_sent / BYTES_PER_SEC
            timeline.rescale_to_audio_time(final_audio_seconds)
            timeline.finalized_index = len(timeline.words) - 1  # Mark all as finalized
            
            final_event = timeline.get_timeline_event()
            final_event["type"] = "FINAL"
            final_event["duration_sec"] = final_audio_seconds
            final_event["total_bytes"] = bytes_sent
            await websocket.send_json(final_event)
            
            logger.info(f"Completed TTS for segment {segment_id}: {final_audio_seconds:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            await websocket.send_json({
                "type": "ERROR",
                "message": str(e),
                "segment_id": segment_id
            })