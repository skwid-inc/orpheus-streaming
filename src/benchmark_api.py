import asyncio
import time
import statistics
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
    
    async def run_single_test(self, session: aiohttp.ClientSession, text: str, voice: str) -> Dict[str, Any]:
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
                    
                    async for chunk in response.content.iter_chunked(4096):
                        if chunk and ttfb is None:
                            ttfb = time.time() - start_time
                        bytes_received += len(chunk)
                    
                    if bytes_received == 0:
                        return {'success': False, 'error': 'No audio data received'}
                    
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
                result = await self.run_single_test(session, request.text, request.voice)
                
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
