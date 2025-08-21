from src.logger import setup_logger
setup_logger()

import time
import os
from src.trt_engine import OrpheusModelTRT
from src.tts_with_timestamps import TTSWithTimestamps
from src.benchmark_api import BenchmarkService, BenchmarkRequest, BenchmarkResult
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
import json
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import warnings
import asyncio

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    input: str = "Hey there, looks like you forgot to provide a prompt!"


class TTSStreamRequest(BaseModel):
    input: str
    continue_: bool = Field(True, alias="continue")
    segment_id: str


engine: OrpheusModelTRT = None
tts_handler: TTSWithTimestamps = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the TTS engine on application startup."""
    global engine, tts_handler
    logger.info("initializing Orpheus")

    engine = OrpheusModelTRT()
    
    
    tts_handler = TTSWithTimestamps(engine)

    logger.info("Orpheus initialized") 
    yield
    # Clean up the model and other resources if needed

app = FastAPI(lifespan=lifespan)


@app.post('/v1/audio/speech/stream')
async def tts_stream(data: TTSRequest):
    """
    Generates audio speech from text in a streaming fashion.
    This endpoint is optimized for low latency (Time to First Byte).
    """
    start_time = time.perf_counter()

    async def generate_audio_stream():
        first_chunk = True
        try:
            audio_generator = engine.generate_speech_async(
                prompt=data.input
            )

            async for chunk in audio_generator:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                    first_chunk = False
                yield chunk
        except Exception:
            logger.exception("An error occurred during audio generation")


    return StreamingResponse(generate_audio_stream(), media_type='audio/pcm')


@app.websocket("/v1/audio/speech/stream/ws")
async def tts_stream_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("connection open")
    try:
        while True:
            data = await websocket.receive_json()

            if not data.get("continue", True):
                logger.info("End of stream message received, closing connection.")
                break

            if not (input_text := data.get("input", "").strip()):
                logger.info("Empty or whitespace-only input received, skipping audio generation.")
                continue

            segment_id = data.get("segment_id", "no_segment_id")

            start_time = time.perf_counter()
            try:
                await websocket.send_json({"type": "start", "segment_id": segment_id})

                if input_text:
                    logger.info(f"Generating audio for input: '{input_text}'")
                    audio_generator = engine.generate_speech_async(
                        prompt=input_text
                    )

                    first_chunk = True
                    async for chunk in audio_generator:
                        if first_chunk:
                            ttfb = time.perf_counter() - start_time
                            logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                            first_chunk = False
                        await websocket.send_bytes(chunk)
                else:
                    logger.info("Empty or whitespace-only input received, skipping audio generation.")
                
                await websocket.send_json({"type": "end", "segment_id": segment_id})

                if not data.get("continue", True):
                    await websocket.send_json({"done": True})
                    break

            except Exception as e:
                logger.exception("An error occurred during audio generation in websocket.")
                await websocket.send_json({"error": str(e), "done": True})
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from websocket.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the websocket endpoint: {e}")
    finally:
        logger.info("Closing websocket connection.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.websocket("/v1/tts")
async def tts_with_timestamps(websocket: WebSocket):
    """WebSocket endpoint with word timestamps support."""
    await tts_handler.handle_websocket(websocket)

@app.post("/v1/benchmark", response_model=BenchmarkResult)
async def run_benchmark(request: BenchmarkRequest):
    """
    Run benchmark tests for TTS generation.
    
    This endpoint performs multiple TTS generation runs and calculates performance metrics
    including TTFB (Time to First Byte), total generation time, and RTF (Real-Time Factor).
    """
    benchmark_service = BenchmarkService(base_url=f"http://localhost:{os.getenv('SERVER_PORT', '9090')}")
    try:
        result = await benchmark_service.run_benchmark(request)
        logger.info(f"Benchmark completed: {result.successful_runs} successful, {result.failed_runs} failed")
        return result
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise