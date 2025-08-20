import asyncio
import aiohttp
import time
import struct
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
import statistics

# Configuration
SERVER_HOST = "localhost"
SERVER_PORT = "9090"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
# Voice parameter removed - model trained with voice=None

# Audio parameters for WAV header generation
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
CHANNELS = 1

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class TTSMetrics:
    connection_time: float
    ttfb: Optional[float]
    streaming_duration: Optional[float]
    total_latency: float
    audio_duration: float
    chunk_count: int
    audio_data: bytes
    bytes_received: int
    success: bool
    error: Optional[str] = None


def generate_wav_header(sample_rate=SAMPLE_RATE, bits_per_sample=BITS_PER_SAMPLE, channels=CHANNELS, data_size=0):
    """Generate WAV header for PCM audio data."""
    bytes_per_sample = bits_per_sample // 8
    block_align = bytes_per_sample * channels
    byte_rate = sample_rate * block_align
    
    # Calculate file size (header + data)
    file_size = 36 + data_size
    
    # Build WAV header
    header = bytearray()
    # RIFF header
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', file_size))
    header.extend(b'WAVE')
    # Format chunk
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))  # Format chunk size
    header.extend(struct.pack('<H', 1))   # PCM format
    header.extend(struct.pack('<H', channels))
    header.extend(struct.pack('<I', sample_rate))
    header.extend(struct.pack('<I', byte_rate))
    header.extend(struct.pack('<H', block_align))
    header.extend(struct.pack('<H', bits_per_sample))
    # Data chunk
    header.extend(b'data')
    header.extend(struct.pack('<I', data_size))
    
    return bytes(header)


async def tts_request(session: aiohttp.ClientSession, text: str):
    """Make a single TTS request using aiohttp for async HTTP."""
    start_time = time.time()
    ttfb = None
    first_chunk_time = None
    last_chunk_time = None
    chunk_count = 0
    
    try:
        # Connection and request timing
        connection_start = time.time()
        
        async with session.post(
            f"{BASE_URL}/v1/audio/speech/stream",
            json={
                "input": text
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            request_sent = time.time()
            connection_time = request_sent - connection_start
            
            if response.status == 200:
                bytes_received = 0
                audio_chunks = []
                
                # Process streaming response
                async for chunk in response.content.iter_chunked(4096):
                    if not chunk:
                        continue
                    
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        ttfb = first_chunk_time - request_sent
                    
                    last_chunk_time = time.time()
                    chunk_count += 1
                    audio_chunks.append(chunk)
                    bytes_received += len(chunk)
                
                end_time = time.time()
                
                # No audio was received
                if bytes_received == 0:
                    return TTSMetrics(
                        connection_time=connection_time,
                        ttfb=None,
                        streaming_duration=None,
                        total_latency=end_time - request_sent,
                        audio_duration=0,
                        chunk_count=0,
                        audio_data=b'',
                        bytes_received=0,
                        success=False,
                        error='No audio data received'
                    )
                
                # Combine all audio chunks
                audio_data = b''.join(audio_chunks)
                
                # Calculate metrics
                streaming_duration = last_chunk_time - first_chunk_time if last_chunk_time and first_chunk_time else None
                total_latency = end_time - request_sent
                
                # Calculate audio duration
                bytes_per_second = SAMPLE_RATE * CHANNELS * (BITS_PER_SAMPLE // 8)
                audio_duration = len(audio_data) / bytes_per_second if bytes_per_second > 0 else 0
                
                return TTSMetrics(
                    connection_time=connection_time,
                    ttfb=ttfb,
                    streaming_duration=streaming_duration,
                    total_latency=total_latency,
                    audio_duration=audio_duration,
                    chunk_count=chunk_count,
                    audio_data=audio_data,
                    bytes_received=bytes_received,
                    success=True
                )
            else:
                error_text = await response.text()
                return TTSMetrics(
                    connection_time=connection_time,
                    ttfb=None,
                    streaming_duration=None,
                    total_latency=time.time() - request_sent,
                    audio_duration=0,
                    chunk_count=0,
                    audio_data=b'',
                    bytes_received=0,
                    success=False,
                    error=f"HTTP {response.status}: {error_text}"
                )
                
    except asyncio.TimeoutError:
        return TTSMetrics(
            connection_time=0,
            ttfb=None,
            streaming_duration=None,
            total_latency=time.time() - start_time,
            audio_duration=0,
            chunk_count=0,
            audio_data=b'',
            bytes_received=0,
            success=False,
            error="Request timeout"
        )
    except Exception as e:
        return TTSMetrics(
            connection_time=0,
            ttfb=None,
            streaming_duration=None,
            total_latency=time.time() - start_time,
            audio_duration=0,
            chunk_count=0,
            audio_data=b'',
            bytes_received=0,
            success=False,
            error=f"Request failed: {e}"
        )


async def run_concurrent_requests(n: int, text: str):
    """Run n concurrent TTS requests."""
    # Create a single session for all requests to enable connection pooling
    async with aiohttp.ClientSession() as session:
        # Warmup request
        print("Running warmup request...")
        warmup_result = await tts_request(session, "Doing warmup")
        if warmup_result.success and warmup_result.ttfb:
            print(f"Warmup complete: TTFB {warmup_result.ttfb:.3f}s")
        else:
            print(f"Warmup failed: {warmup_result.error}")
        
        # Run concurrent requests
        tasks = [tts_request(session, text) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        
    return results


def save_audio(audio_data: bytes, filename: str):
    """Save audio data as WAV file."""
    wav_header = generate_wav_header(data_size=len(audio_data))
    with open(filename, 'wb') as f:
        f.write(wav_header)
        f.write(audio_data)


def print_metrics(results: list[TTSMetrics], concurrency_level: int):
    """Print detailed metrics for the benchmark results."""
    print(f"\n=== Concurrency Level: {concurrency_level} ===")
    
    # Filter successful results
    successful_results = [r for r in results if r.success]
    failed_count = len(results) - len(successful_results)
    
    if failed_count > 0:
        print(f"Failed requests: {failed_count}/{len(results)}")
        for i, r in enumerate(results):
            if not r.success:
                print(f"  Request {i+1}: {r.error}")
    
    if not successful_results:
        print("No successful requests to analyze.")
        return
    
    # Extract metrics from successful results
    ttfb_values = [r.ttfb for r in successful_results if r.ttfb is not None]
    total_latencies = [r.total_latency for r in successful_results]
    audio_durations = [r.audio_duration for r in successful_results]
    chunk_counts = [r.chunk_count for r in successful_results]
    streaming_durations = [r.streaming_duration for r in successful_results if r.streaming_duration is not None]
    
    print(f"\nSuccessful requests: {len(successful_results)}/{len(results)}")
    
    if ttfb_values:
        print("\nTTFB (Time To First Byte):")
        print(f"  Mean: {np.mean(ttfb_values):.3f}s")
        print(f"  Median: {np.median(ttfb_values):.3f}s")
        print(f"  Min: {np.min(ttfb_values):.3f}s")
        print(f"  Max: {np.max(ttfb_values):.3f}s")
        print(f"  StdDev: {np.std(ttfb_values):.3f}s")
        if len(ttfb_values) >= 10:
            print(f"  P95: {np.percentile(ttfb_values, 95):.3f}s")
            print(f"  P99: {np.percentile(ttfb_values, 99):.3f}s")
    
    print(f"\nTotal Generation Time:")
    print(f"  Mean: {np.mean(total_latencies):.3f}s")
    print(f"  Median: {np.median(total_latencies):.3f}s")
    print(f"  Min: {np.min(total_latencies):.3f}s")
    print(f"  Max: {np.max(total_latencies):.3f}s")
    
    if streaming_durations:
        print(f"\nStreaming Duration (after first byte):")
        print(f"  Mean: {np.mean(streaming_durations):.3f}s")
    
    print(f"\nAudio Characteristics:")
    print(f"  Sample Rate: {SAMPLE_RATE}Hz, {BITS_PER_SAMPLE}-bit, {CHANNELS} channel(s)")
    print(f"  Average Audio Duration: {np.mean(audio_durations):.2f}s")
    print(f"  Average Chunks: {np.mean(chunk_counts):.1f}")
    
    # Calculate real-time factor (RTF)
    rtf_values = [total_lat / audio_dur for total_lat, audio_dur in zip(total_latencies, audio_durations) if audio_dur > 0]
    if rtf_values:
        print(f"\nReal-time Factor (RTF):")
        print(f"  Mean: {np.mean(rtf_values):.3f}")
        print(f"  Min: {np.min(rtf_values):.3f}")
        print(f"  Max: {np.max(rtf_values):.3f}")
        print(f"  (Lower is better; RTF < 1 means faster than real-time)")


def aggregate_metrics(all_results: list[list[TTSMetrics]], concurrency_level: int):
    """Aggregate metrics from multiple runs."""
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS - Concurrency: {concurrency_level}, Runs: {len(all_results)}")
    print(f"{'='*80}")
    
    # Flatten all successful results
    all_successful = []
    total_requests = 0
    for run_results in all_results:
        total_requests += len(run_results)
        all_successful.extend([r for r in run_results if r.success])
    
    if not all_successful:
        print("No successful requests across all runs!")
        return
    
    success_rate = (len(all_successful) / total_requests) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}% ({len(all_successful)}/{total_requests})")
    
    # Extract all metrics
    ttfb_values = [r.ttfb for r in all_successful if r.ttfb is not None]
    total_latencies = [r.total_latency for r in all_successful]
    audio_durations = [r.audio_duration for r in all_successful]
    rtf_values = [lat/dur for lat, dur in zip(total_latencies, audio_durations) if dur > 0]
    
    if ttfb_values:
        print("\nTTFB (Time To First Byte) - All Runs:")
        print(f"  Mean: {np.mean(ttfb_values):.3f}s")
        print(f"  Median: {np.median(ttfb_values):.3f}s")
        print(f"  Min: {np.min(ttfb_values):.3f}s")
        print(f"  Max: {np.max(ttfb_values):.3f}s")
        print(f"  StdDev: {np.std(ttfb_values):.3f}s")
        print(f"  P50: {np.percentile(ttfb_values, 50):.3f}s")
        print(f"  P95: {np.percentile(ttfb_values, 95):.3f}s")
        print(f"  P99: {np.percentile(ttfb_values, 99):.3f}s")
    
    print(f"\nTotal Generation Time - All Runs:")
    print(f"  Mean: {np.mean(total_latencies):.3f}s")
    print(f"  Median: {np.median(total_latencies):.3f}s")
    print(f"  Min: {np.min(total_latencies):.3f}s")
    print(f"  Max: {np.max(total_latencies):.3f}s")
    print(f"  StdDev: {np.std(total_latencies):.3f}s")
    
    if rtf_values:
        print(f"\nReal-time Factor (RTF) - All Runs:")
        print(f"  Mean: {np.mean(rtf_values):.3f}")
        print(f"  Median: {np.median(rtf_values):.3f}")
        print(f"  Min: {np.min(rtf_values):.3f}")
        print(f"  Max: {np.max(rtf_values):.3f}")
        print(f"  P95: {np.percentile(rtf_values, 95):.3f}")
        print(f"  (Lower is better; RTF < 1 means faster than real-time)")
    
    # Per-run summary
    print(f"\nPer-Run TTFB Summary:")
    for i, run_results in enumerate(all_results):
        run_ttfb = [r.ttfb for r in run_results if r.success and r.ttfb is not None]
        if run_ttfb:
            print(f"  Run {i+1}: Mean={np.mean(run_ttfb):.3f}s, Min={np.min(run_ttfb):.3f}s, Max={np.max(run_ttfb):.3f}s")


def main():
    """Main benchmark function."""
    print("TTS HTTP Streaming Latency Profiler")
    print(f"Server: {BASE_URL}")

    
    # Configuration
    CONCURRENCY = 4
    NUM_RUNS = 10
    SLEEP_BETWEEN_RUNS = 2.0  # seconds
    
    print(f"\nConfiguration:")
    print(f"  Concurrency Level: {CONCURRENCY}")
    print(f"  Number of Runs: {NUM_RUNS}")
    print(f"  Sleep Between Runs: {SLEEP_BETWEEN_RUNS}s")
    
    # Test text - same as original
    text = "How are you doing today? So happy to see you again today for the second time."
    
    # Create timestamp for this benchmark session
    from datetime import datetime
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"{OUTPUT_DIR}/benchmark_{session_timestamp}"
    os.makedirs(session_dir, exist_ok=True)
    print(f"  Output Directory: {session_dir}")
    
    # Collect results from all runs
    all_results = []
    
    for run_num in range(1, NUM_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"RUN {run_num}/{NUM_RUNS} - Concurrency: {CONCURRENCY}")
        print(f"{'='*60}")
        
        # Run the concurrent requests
        results = asyncio.run(run_concurrent_requests(CONCURRENCY, text))
        all_results.append(results)
        
        # Print metrics for this run
        print_metrics(results, CONCURRENCY)
        
        # Save audio files from this run
        saved_count = 0
        for i, result in enumerate(results):
            if result.success and result.audio_data:
                filename = f"{session_dir}/run{run_num:02d}_req{i+1}_concurrent{CONCURRENCY}.wav"
                save_audio(result.audio_data, filename)
                saved_count += 1
        
        if saved_count > 0:
            print(f"\nSaved {saved_count} audio files for run {run_num}")
        
        # Sleep between runs (except after last run)
        if run_num < NUM_RUNS:
            print(f"\nSleeping {SLEEP_BETWEEN_RUNS}s before next run...")
            time.sleep(SLEEP_BETWEEN_RUNS)
    
    # Print aggregate results
    aggregate_metrics(all_results, CONCURRENCY)
    
    print(f"\n{'='*80}")
    print(f"All audio files saved to: {session_dir}")
    print(f"Total files: {len([f for f in os.listdir(session_dir) if f.endswith('.wav')])}")


if __name__ == "__main__":
    main()