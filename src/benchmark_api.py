import asyncio
import time
import statistics
import os
import struct
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import aiohttp
import logging

logger = logging.getLogger(__name__)

# Audio parameters
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
CHANNELS = 1


class BenchmarkRequest(BaseModel):
    text: str = Field(..., description="Text to benchmark TTS generation")
    voice: str = Field(..., description="Voice to use for generation")
    num_runs: int = Field(5, ge=1, le=20, description="Number of benchmark runs")
    warmup: bool = Field(True, description="Whether to perform a warmup run")
    warmup_text: Optional[str] = Field("Doing warmup", description="Text for warmup run")


class BenchmarkResult(BaseModel):
    successful_runs: int
    failed_runs: int
    text_length: int
    voice: str
    metrics: Dict[str, Any]


class BenchmarkService:
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.tts_url = f"{base_url}/v1/audio/speech/stream"
        self.output_dir = "api_benchmark_outputs"
        self.saved_files = []
    
    def generate_wav_header(self, data_size: int) -> bytes:
        """Generate WAV header for PCM audio data."""
        bytes_per_sample = BITS_PER_SAMPLE // 8
        block_align = bytes_per_sample * CHANNELS
        byte_rate = SAMPLE_RATE * block_align
        
        file_size = 36 + data_size
        
        header = bytearray()
        header.extend(b'RIFF')
        header.extend(struct.pack('<I', file_size))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend(struct.pack('<I', 16))
        header.extend(struct.pack('<H', 1))
        header.extend(struct.pack('<H', CHANNELS))
        header.extend(struct.pack('<I', SAMPLE_RATE))
        header.extend(struct.pack('<I', byte_rate))
        header.extend(struct.pack('<H', block_align))
        header.extend(struct.pack('<H', BITS_PER_SAMPLE))
        header.extend(b'data')
        header.extend(struct.pack('<I', data_size))
        
        return bytes(header)
    
    async def run_single_test(self, session: aiohttp.ClientSession, text: str, voice: str, save_file: Optional[str] = None) -> Dict[str, Any]:
        """Run a single TTS test and return timing metrics."""
        start_time = time.time()
        ttfb = None
        
        try:
            async with session.post(
                self.tts_url,
                json={"input": text, "voice": voice},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    bytes_received = 0
                    audio_data = bytearray()
                    
                    async for chunk in response.content.iter_chunked(4096):
                        if chunk and ttfb is None:
                            ttfb = time.time() - start_time
                        audio_data.extend(chunk)
                        bytes_received += len(chunk)
                    
                    if bytes_received == 0:
                        return {'success': False, 'error': 'No audio data received'}
                    
                    # Save audio file if requested
                    if save_file:
                        wav_header = self.generate_wav_header(len(audio_data))
                        with open(save_file, 'wb') as f:
                            f.write(wav_header)
                            f.write(audio_data)
                        self.saved_files.append(save_file)
                    
                    total_time = time.time() - start_time
                    
                    # Calculate audio duration from PCM data
                    bytes_per_second = SAMPLE_RATE * CHANNELS * (BITS_PER_SAMPLE // 8)
                    audio_duration = bytes_received / bytes_per_second if bytes_per_second > 0 else 0
                    
                    return {
                        'success': True,
                        'ttfb': ttfb,
                        'total_time': total_time,
                        'bytes_received': bytes_received,
                        'audio_duration': audio_duration
                    }
                else:
                    error_text = await response.text()
                    return {'success': False, 'error': f"HTTP {response.status}: {error_text}"}
                    
        except asyncio.TimeoutError:
            return {'success': False, 'error': 'Request timeout'}
        except Exception as e:
            return {'success': False, 'error': f"Request failed: {str(e)}"}
    
    async def run_benchmark(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Run complete benchmark test."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.saved_files = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Warmup run
                if request.warmup:
                    logger.info("Running warmup...")
                    warmup_result = await self.run_single_test(session, request.warmup_text, request.voice)
                    if warmup_result['success']:
                        logger.info(f"Warmup complete: TTFB {warmup_result['ttfb']:.3f}s")
                    else:
                        logger.warning(f"Warmup failed: {warmup_result['error']}")
                
                # Benchmark runs
                logger.info(f"Starting {request.num_runs} benchmark runs...")
                results = []
                failed_runs = 0
                
                for run in range(1, request.num_runs + 1):
                    logger.info(f"Running benchmark {run}/{request.num_runs}...")
                    save_path = os.path.join(self.output_dir, f"run_{run}.wav")
                    result = await self.run_single_test(session, request.text, request.voice, save_path)
                    
                    if result['success']:
                        results.append(result)
                    else:
                        failed_runs += 1
                        logger.error(f"Run {run} failed: {result['error']}")
            
            # Calculate statistics
            metrics = {}
            
            if results:
                ttfb_times = [r['ttfb'] for r in results if r['ttfb'] is not None]
                total_times = [r['total_time'] for r in results]
                audio_durations = [r['audio_duration'] for r in results if r.get('audio_duration')]
                rtfs = [r['total_time'] / r['audio_duration'] for r in results if r.get('audio_duration', 0) > 0]
                
                if audio_durations:
                    metrics['audio'] = {
                        'sample_rate': SAMPLE_RATE,
                        'bits_per_sample': BITS_PER_SAMPLE,
                        'channels': CHANNELS,
                        'average_duration_seconds': round(statistics.mean(audio_durations), 3)
                    }
                
                if ttfb_times:
                    metrics['ttfb'] = {
                        'mean_ms': round(statistics.mean(ttfb_times) * 1000, 2),
                        'stddev_ms': round(statistics.stdev(ttfb_times) * 1000, 2) if len(ttfb_times) > 1 else 0,
                        'min_ms': round(min(ttfb_times) * 1000, 2),
                        'max_ms': round(max(ttfb_times) * 1000, 2)
                    }
                
                if total_times:
                    metrics['total_time'] = {
                        'mean_seconds': round(statistics.mean(total_times), 3),
                        'stddev_seconds': round(statistics.stdev(total_times), 3) if len(total_times) > 1 else 0,
                        'min_seconds': round(min(total_times), 3),
                        'max_seconds': round(max(total_times), 3)
                    }
                
                if rtfs:
                    metrics['rtf'] = {
                        'mean': round(statistics.mean(rtfs), 3),
                        'stddev': round(statistics.stdev(rtfs), 3) if len(rtfs) > 1 else 0,
                        'min': round(min(rtfs), 3),
                        'max': round(max(rtfs), 3),
                        'description': 'Real-Time Factor (lower is better, <1 means faster than real-time)'
                    }
            
            return BenchmarkResult(
                successful_runs=len(results),
                failed_runs=failed_runs,
                text_length=len(request.text),
                voice=request.voice,
                metrics=metrics
            )
        
        finally:
            # Clean up saved files and directory
            if os.path.exists(self.output_dir):
                logger.info(f"Cleaning up {self.output_dir} directory...")
                try:
                    shutil.rmtree(self.output_dir)
                    logger.info(f"Successfully deleted {self.output_dir}")
                except Exception as e:
                    logger.error(f"Failed to delete {self.output_dir}: {e}")
