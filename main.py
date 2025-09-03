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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from starlette.websockets import WebSocketState
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import warnings
import asyncio
import uuid
import re

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


@app.get("/health")
async def health():
    return {"status": "ok"}

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

@app.post("/production/predict")
async def baseten_predict(model_input: Dict[str, Any], request: Request) -> StreamingResponse:
    """
    Baseten-compatible prediction endpoint that matches the model.py API.
    This endpoint allows the server to be called using the same interface as call.py.
    """
    try:
        # Extract parameters from model_input
        req_id = str(model_input.get("request_id", uuid.uuid4()))
        raw_prompt = str(model_input.get("prompt", ""))
        voice = model_input.get("voice", "tara")
        
        # Check for non-speakable content (similar to model.py)
        _NON_SPEAKABLE_RE = re.compile(r"^\s*[-–—.,;:!?…·•*#_/\\()\[\]{}'\"|~`^]+\s*$")
        
        def _is_speakable(text_val: str) -> bool:
            if not text_val or not text_val.strip():
                return False
            if _NON_SPEAKABLE_RE.match(text_val):
                return False
            return any(ch.isalnum() for ch in text_val)
        
        if not _is_speakable(raw_prompt):
            logger.info(f"Excluding non-speakable prompt for request_id {req_id}: '{raw_prompt}'")
            return Response(status_code=204)  # No content for non-speakable prompts
        
        # Format prompt with voice (simplified version - your engine might handle this differently)
        formatted_prompt = raw_prompt
        if voice and voice != "None":
            # You might need to adapt this based on how your engine handles voices
            formatted_prompt = f"{voice}: {raw_prompt}"
        
        input_length = len(formatted_prompt)
        MAX_CHARACTERS_INPUT = 6144  # Match model.py's limit
        
        logger.info(f"Starting request_id {req_id} with input length {input_length}")
        
        if input_length > MAX_CHARACTERS_INPUT:
            return Response(
                content=(
                    f"Your suggested prompt is too long (len: {input_length}), "
                    f"max length is {MAX_CHARACTERS_INPUT} characters. "
                    "To generate audio faster, please split your request into multiple prompts."
                ),
                status_code=400
            )
        
        start_time = time.perf_counter()
        
        async def generate_audio_stream():
            """Generate audio stream in the format expected by call.py (raw PCM audio)."""
            try:
                first_chunk = True
                # Use your existing engine to generate audio
                audio_generator = engine.generate_speech_async(prompt=formatted_prompt)
                
                async for chunk in audio_generator:
                    if first_chunk:
                        ttfb = time.perf_counter() - start_time
                        logger.info(f"[{req_id}] Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                        first_chunk = False
                    # The chunk should already be raw PCM audio bytes from your engine
                    yield chunk
                
                total_time = time.perf_counter() - start_time
                logger.info(f"[{req_id}] Finished generation in {total_time:.2f}s")
                
            except Exception as e:
                logger.exception(f"Error in request_id {req_id} during audio generation: {e}")
                # In streaming, we can't change status code, so we just stop
                return
        
        # Return streaming response with audio/wav media type to match model.py
        return StreamingResponse(
            generate_audio_stream(),
            media_type="audio/wav",  # Match model.py's media type
            headers={
                "X-Baseten-Input-Tokens": str(input_length),
                "X-Request-Id": req_id
            }
        )
        
    except Exception as e:
        logger.error(f"Error in request_id {req_id}: {e} with input {model_input}")
        return Response(
            content=f"An internal server error occurred while processing your request {req_id}",
            status_code=500
        )