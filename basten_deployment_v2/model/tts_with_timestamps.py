"""
FastAPI WebSocket endpoint with word timestamps for Orpheus.
"""

import asyncio
import json
import time
import logging
import uuid
import re
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from typing import Optional, AsyncGenerator

from model.word_timestamps import WordTimestampGenerator

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
        client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        connection_start = time.time()
        total_bytes_sent = 0
        total_requests = 0
        
        logger.info(f"WebSocket connection opened with timestamps support - client: {client_info}")
        
        # Per-connection segment state
        segment_meta_sent: set[str] = set()
        segment_stats: dict[str, dict] = {}
        last_segment_id: str | None = None
        
        # Send capabilities for basic format negotiation and feature discovery
        capabilities = {
            "type": "CAPABILITIES",
            "version": "1.0",
            "audio": {
                "formats": ["pcm_s16le"],
                "sample_rates": [SR],
                "channels": [CHANNELS],
                "word_timestamps": True,
                "streaming": True
            },
            "features": {
                "progress_updates": True,
                "heartbeat": True,
                "context_id": True
            }
        }
        try:
            await websocket.send_json(capabilities)
        except Exception:
            logger.debug("Failed to send CAPABILITIES on connect")
        
        try:
            # Heartbeat sender to keep connection healthy and detectable
            async def heartbeat_sender():
                try:
                    while True:
                        await asyncio.sleep(15)
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_json({
                                "type": "HEARTBEAT",
                                "timestamp": time.time()
                            })
                        else:
                            break
                except Exception:
                    # Silently end heartbeat on errors (likely connection closing)
                    return

            hb_task = asyncio.create_task(heartbeat_sender())

            while True:
                # Receive request (typed or legacy), with timeout to allow heartbeats
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                except asyncio.TimeoutError:
                    # No message, continue to send heartbeat
                    continue

                total_requests += 1

                # New typed protocol support
                msg_type = data.get("type")
                if msg_type:
                    if msg_type == "PING":
                        await websocket.send_json({"type": "PONG", "timestamp": time.time()})
                        continue
                    if msg_type == "STOP":
                        logger.info(f"Client {client_info}: STOP received, closing connection")
                        break
                    if msg_type == "CONFIGURE":
                        # Currently only speaking_rate supported
                        logger.info(f"Client {client_info}: CONFIGURE - {data}")
                        await websocket.send_json({"type": "CONFIGURED"})
                        continue
                    if msg_type == "SYNTHESIZE":
                        text = (data.get("text") or data.get("input") or "").strip()
                        if not text:
                            await websocket.send_json({"type": "ERROR", "message": "No text provided"})
                            continue
                        speaking_rate = data.get("speaking_rate", "normal")
                        segment_id = data.get("segment_id", "default")
                        context_id = data.get("context_id", "default_context")
                        request_id = data.get("request_id") or f"req_{uuid.uuid4()}"

                        logger.info(
                            f"Client {client_info}: SYNTHESIZE #{total_requests} ctx={context_id} seg={segment_id} len={len(text)}"
                        )

                        # Notify start
                        await websocket.send_json({
                            "type": "SYNTHESIS_STARTED",
                            "request_id": request_id,
                            "segment_id": segment_id,
                            "context_id": context_id,
                            "text_length": len(text)
                        })

                        # Process the request
                        bytes_sent = await self._process_tts_request(
                            websocket, text, speaking_rate, segment_id, client_info,
                            context_id=context_id, request_id=request_id,
                            segment_meta_sent=segment_meta_sent, segment_stats=segment_stats,
                            is_final=True
                        )
                        total_bytes_sent += bytes_sent
                        last_segment_id = segment_id
                        continue

                    # Unknown typed message
                    await websocket.send_json({"type": "ERROR", "message": f"Unknown type: {msg_type}"})
                    continue

                # Legacy schema support
                if not data.get("continue", True):
                    # End-of-input marker: emit a single FINAL for the last segment seen
                    if last_segment_id and last_segment_id in segment_stats:
                        stats = segment_stats[last_segment_id]
                        total_bytes = stats.get("bytes_total", 0)
                        logger.debug(f"Client {client_info}: End-of-input for segment {last_segment_id}, stats: {stats}")
                        final_event = {
                            "type": "FINAL",
                            "segment_id": last_segment_id,
                            "context_id": last_segment_id,  # Cartesia expects context_id; using segment_id
                            "words": stats.get("last_words", []),
                            "start": stats.get("last_start", []),
                            "end": stats.get("last_end", []),
                            "total_bytes": total_bytes,
                            "duration_sec": total_bytes / BYTES_PER_SEC if BYTES_PER_SEC else 0.0,
                        }
                        try:
                            await websocket.send_json(final_event)
                        except Exception:
                            logger.debug("Failed to send FINAL event")
                        logger.info(f"Client {client_info}: Emitted FINAL for segment {last_segment_id} - total bytes {total_bytes}")
                    else:
                        logger.info(f"Client {client_info}: End of stream received with no active segment")
                    # Do not necessarily close; client may close
                    break
                text = data.get("text", data.get("input", "")).strip()
                if not text:
                    logger.info(f"Client {client_info}: Empty text received in request #{total_requests}, skipping")
                    continue
                speaking_rate = data.get("speaking_rate", "normal")
                segment_id = data.get("segment_id", "default")
                logger.info(f"Client {client_info}: Received TTS request #{total_requests} - text: '{text}' (length: {len(text)} chars), segment_id: {segment_id}, speaking_rate: {speaking_rate}")
                # Log existing segment stats if this segment_id was used before
                if segment_id in segment_stats:
                    logger.warning(f"Client {client_info}: Segment ID {segment_id} reused! Existing stats: {segment_stats[segment_id]}")
                bytes_sent = await self._process_tts_request(
                    websocket, text, speaking_rate, segment_id, client_info,
                    segment_meta_sent=segment_meta_sent, segment_stats=segment_stats,
                    is_final=False
                )
                total_bytes_sent += bytes_sent
                last_segment_id = segment_id
                
        except WebSocketDisconnect:
            logger.info(f"Client {client_info}: Disconnected from WebSocket")
        except Exception as e:
            logger.error(f"Client {client_info}: WebSocket error: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "ERROR", "message": str(e)})
        finally:
            try:
                hb_task.cancel()
            except Exception:
                pass
            connection_duration = time.time() - connection_start
            logger.info(f"Client {client_info}: Closing WebSocket connection - duration: {connection_duration:.2f}s, total requests: {total_requests}, total bytes sent: {total_bytes_sent:,}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
    
    async def _process_tts_request(self, websocket: WebSocket, text: str, 
                                  speaking_rate: str, segment_id: str, client_info: str,
                                  context_id: str | None = None, request_id: str | None = None,
                                  segment_meta_sent: set[str] | None = None,
                                  segment_stats: dict[str, dict] | None = None,
                                  is_final: bool = True):
        """Process a single TTS request with timestamps."""
        
        logger.info(f"Client {client_info}: Starting TTS processing for segment {segment_id}")
        
        # Send metadata
        effective_context_id = context_id or segment_id
        if segment_meta_sent is None or segment_id not in segment_meta_sent:
            await websocket.send_json({
                "type": "META",
                "sample_rate": SR,
                "channels": CHANNELS,
                "format": "pcm_s16le",
                "segment_id": segment_id,
                "context_id": effective_context_id,
                **({"request_id": request_id} if request_id else {})
            })
            if segment_meta_sent is not None:
                segment_meta_sent.add(segment_id)
        logger.info(f"Client {client_info}: Sent audio metadata for segment {segment_id}")
        
        # Speakability helpers
        _NON_SPEAKABLE_RE = re.compile(r"^\s*[-–—.,;:!?…·•*#_/\\()\[\]{}'\"|~`^]+\s*$")

        def is_speakable(text_val: str) -> bool:
            if not text_val or not text_val.strip():
                return False
            if _NON_SPEAKABLE_RE.match(text_val):
                return False
            return any(ch.isalnum() for ch in text_val)

        # Non-speakable fast-path
        if not is_speakable(text):
            empty = {
                "type": "timestamps",
                "segment_id": segment_id,
                "context_id": effective_context_id,
                **({"request_id": request_id} if request_id else {}),
                "word_timestamps": {
                    "words": [],
                    "start": [],
                    "end": []
                }
            }
            await websocket.send_json(empty)

            # Track for legacy flow FINAL at end-of-input
            if segment_stats is not None:
                st = segment_stats.setdefault(segment_id, {})
                st["bytes_total"] = 0  # Explicitly set to 0 for non-speakable
                st["last_words"] = []
                st["last_start"] = []
                st["last_end"] = []
                logger.debug(f"Client {client_info}: Non-speakable text for segment {segment_id}, set bytes_total=0")

            if is_final:
                final_event = {
                    "type": "FINAL",
                    "segment_id": segment_id,
                    "context_id": effective_context_id,
                    **({"request_id": request_id} if request_id else {}),
                    "words": [],
                    "start": [],
                    "end": [],
                    "duration_sec": 0.0,
                    "total_bytes": 0
                }
                await websocket.send_json(final_event)

                if request_id or context_id:
                    await websocket.send_json({
                        "type": "SYNTHESIS_COMPLETED",
                        **({"request_id": request_id} if request_id else {}),
                        "context_id": effective_context_id,
                        "segment_id": segment_id,
                        "stats": {
                            "generation_time_ms": 0.0,
                            "audio_duration_s": 0.0,
                            "total_bytes": 0,
                            "chunks": 0,
                            "rtf": 0.0,
                        }
                    })
            return 0

        # Create timeline
        timeline = self.timestamp_gen.create_timeline(text, speaking_rate)
        
        if not timeline:
            logger.warning(f"Client {client_info}: Failed to create timeline for segment {segment_id}")
            await websocket.send_json({"type": "ERROR", "message": "Failed to create timeline"})
            return 0
        
        # Helper: normalize words by removing commas and question marks; drop empty tokens
        def normalize_word_timestamps(words_list, start_list, end_list):
            norm_words = []
            norm_start = []
            norm_end = []
            for w, s, e in zip(words_list, start_list, end_list):
                cw = w.replace(",", "").replace("?", "")
                if cw.strip():
                    norm_words.append(cw)
                    norm_start.append(s)
                    norm_end.append(e)
            return norm_words, norm_start, norm_end

        # Send initial timeline (non-final) as Cartesia-compatible timestamps event
        initial_src = timeline.get_timeline_event()
        _w, _s, _e = (
            initial_src.get("words", []),
            initial_src.get("start", []),
            initial_src.get("end", []),
        )
        _w, _s, _e = normalize_word_timestamps(_w, _s, _e)
        initial_event = {
            "type": "timestamps",
            "segment_id": segment_id,
            "context_id": effective_context_id,
            **({"request_id": request_id} if request_id else {}),
            "word_timestamps": {
                "words": _w,
                "start": _s,
                "end": _e,
            }
        }
        await websocket.send_json(initial_event)
        logger.info(
            f"TS seg={segment_id} words={len(_w)} preview={_w[:3]}...{_w[-3:] if len(_w)>3 else []}"
        )
        if segment_stats is not None:
            st = segment_stats.setdefault(segment_id, {"bytes_total": 0})
            st["last_words"] = list(initial_event["word_timestamps"]["words"])
            st["last_start"] = list(initial_event["word_timestamps"]["start"])
            st["last_end"] = list(initial_event["word_timestamps"]["end"])
        logger.info(f"Client {client_info}: Sent initial timeline with {len(timeline.words)} words for segment {segment_id}")
        
        # Start synthesis
        start_time = time.perf_counter()
        bytes_sent = 0
        chunk_count = 0
        last_rescale_time = 0.0
        rescale_interval = timeline.voice_profile.rescale_interval_ms / 1000.0
        timeline_updates = 0
        
        try:
            # Generate audio
            logger.info(f"Client {client_info}: Starting audio generation for segment {segment_id}")
            audio_generator = self.engine.generate_speech_async(
                prompt=text
            )
            
            first_chunk = True
            async for audio_chunk in audio_generator:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    logger.info(f"Client {client_info}: TTFB: {ttfb*1000:.2f}ms for segment {segment_id}")
                    if request_id or context_id:
                        await websocket.send_json({
                            "type": "FIRST_AUDIO_CHUNK",
                            **({"request_id": request_id} if request_id else {}),
                            "context_id": effective_context_id,
                            "ttfb_ms": ttfb * 1000
                        })
                    first_chunk = False
                
                # Send audio chunk
                await websocket.send_bytes(audio_chunk)
                chunk_count += 1
                bytes_sent += len(audio_chunk)
                
                # Calculate audio time
                audio_seconds = bytes_sent / BYTES_PER_SEC
                
                # Rescale timeline periodically
                if audio_seconds - last_rescale_time >= rescale_interval:
                    if timeline.rescale_to_audio_time(audio_seconds):
                        src = timeline.get_timeline_event()
                        _w, _s, _e = normalize_word_timestamps(
                            src.get("words", []), src.get("start", []), src.get("end", [])
                        )
                        event = {
                            "type": "timestamps",
                            "segment_id": segment_id,
                            "context_id": effective_context_id,
                            **({"request_id": request_id} if request_id else {}),
                            "word_timestamps": {"words": _w, "start": _s, "end": _e}
                        }
                        await websocket.send_json(event)
                        logger.info(
                            f"TS seg={segment_id} words={len(_w)} preview={_w[:3]}...{_w[-3:] if len(_w)>3 else []}"
                        )
                        if segment_stats is not None:
                            st = segment_stats.setdefault(segment_id, {"bytes_total": 0})
                            st["last_words"] = list(_w)
                            st["last_start"] = list(_s)
                            st["last_end"] = list(_e)
                        timeline_updates += 1
                        logger.debug(f"Client {client_info}: Sent timeline update #{timeline_updates} at {audio_seconds:.2f}s for segment {segment_id}")
                    last_rescale_time = audio_seconds

                # Periodic progress updates
                if (request_id or context_id) and (chunk_count % 10 == 0):
                    await websocket.send_json({
                        "type": "PROGRESS",
                        **({"request_id": request_id} if request_id else {}),
                        "context_id": effective_context_id,
                        "segment_id": segment_id,
                        "chunks_sent": chunk_count,
                        "bytes_sent": bytes_sent
                    })
            
            # Finalization or accumulation
            if segment_stats is not None:
                st = segment_stats.setdefault(segment_id, {"bytes_total": 0})
                st["bytes_total"] = st.get("bytes_total", 0) + bytes_sent
            
            final_audio_seconds = bytes_sent / BYTES_PER_SEC
            timeline.rescale_to_audio_time(final_audio_seconds)
            timeline.finalized_index = len(timeline.words) - 1  # Mark all as finalized
            
            if is_final:
                # Optionally send one last timestamps before FINAL
                src = timeline.get_timeline_event()
                _w, _s, _e = normalize_word_timestamps(
                    src.get("words", []), src.get("start", []), src.get("end", [])
                )
                last_ts = {
                    "type": "timestamps",
                    "segment_id": segment_id,
                    "context_id": effective_context_id,
                    **({"request_id": request_id} if request_id else {}),
                    "word_timestamps": {"words": _w, "start": _s, "end": _e}
                }
                await websocket.send_json(last_ts)
                logger.info(
                    f"TS seg={segment_id} words={len(_w)} preview={_w[:3]}...{_w[-3:] if len(_w)>3 else []}"
                )

                # Send minimal FINAL
                total_bytes_final = bytes_sent
                if segment_stats is not None:
                    total_bytes_final = segment_stats.get(segment_id, {}).get("bytes_total", bytes_sent)
                final_event = {
                    "type": "FINAL",
                    "segment_id": segment_id,
                    "context_id": effective_context_id,
                    **({"request_id": request_id} if request_id else {}),
                    "duration_sec": total_bytes_final / BYTES_PER_SEC if BYTES_PER_SEC else 0.0,
                    "total_bytes": total_bytes_final
                }
                await websocket.send_json(final_event)
                logger.info(
                    f"FINAL seg={segment_id} bytes={total_bytes_final} dur={final_event['duration_sec']:.2f}s"
                )
            else:
                # Not final: update last_timeline snapshot for later FINAL emission
                snapshot = timeline.get_timeline_event()
                if segment_stats is not None:
                    st = segment_stats.setdefault(segment_id, {"bytes_total": 0})
                    st["last_words"] = list(snapshot.get("words", []))
                    st["last_start"] = list(snapshot.get("start", []))
                    st["last_end"] = list(snapshot.get("end", []))
            
            generation_time = time.perf_counter() - start_time
            logger.info(f"Client {client_info}: Completed segment {segment_id} - sent {bytes_sent:,} bytes in {chunk_count} chunks, duration: {final_audio_seconds:.2f}s, generation time: {generation_time*1000:.2f}ms, timeline updates: {timeline_updates}")
            
            # Completion summary (best-practice event)
            if (request_id or context_id) and is_final:
                rtf = (generation_time / final_audio_seconds) if final_audio_seconds > 0 else 0.0
                await websocket.send_json({
                    "type": "SYNTHESIS_COMPLETED",
                    **({"request_id": request_id} if request_id else {}),
                    "context_id": effective_context_id,
                    "segment_id": segment_id,
                    "stats": {
                        "generation_time_ms": generation_time * 1000,
                        "audio_duration_s": (segment_stats.get(segment_id, {}).get("bytes_total", bytes_sent) / BYTES_PER_SEC) if segment_stats is not None else final_audio_seconds,
                        "total_bytes": segment_stats.get(segment_id, {}).get("bytes_total", bytes_sent) if segment_stats is not None else bytes_sent,
                        "chunks": chunk_count,
                        "rtf": rtf
                    }
                })
            
            return bytes_sent
            
        except Exception as e:
            logger.error(f"Client {client_info}: Error during synthesis for segment {segment_id}: {e}")
            await websocket.send_json({
                "type": "ERROR",
                "message": str(e),
                "segment_id": segment_id,
                **({"context_id": context_id} if context_id else {}),
                **({"request_id": request_id} if request_id else {})
            })
            return bytes_sent