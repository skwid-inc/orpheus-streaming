from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import os
import time
import logging
from collections import deque, defaultdict

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(snac_device)

# Profiling setup for decoder
_decoder_profiling_data = defaultdict(list)
_decoder_profiling_enabled = os.getenv("ENABLE_PROFILING", "true").lower() == "true"
logger = logging.getLogger(__name__)

# Tensor buffer cache for reuse (avoids repeated allocation)
_tensor_buffer_cache = {}
_max_cached_frames = 16  # Cache tensors for up to 16 frames

class DecoderProfilerTimer:
    """Context manager for timing decoder operations."""
    
    def __init__(self, operation_name, enabled=True):
        self.operation_name = operation_name
        self.enabled = enabled and _decoder_profiling_enabled
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        if self.enabled:
            self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.start_time:
            self.end_time = time.perf_counter()
            duration = self.end_time - self.start_time
            _decoder_profiling_data[self.operation_name].append(duration)
            logger.info(f"🎵 {self.operation_name}: {duration*1000:.2f}ms")
            
    def get_duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

def _get_cached_tensors(num_frames):
    """
    Get cached tensors for the given number of frames, or create new ones.
    This avoids repeated tensor allocation for common frame counts.
    """
    if num_frames > _max_cached_frames:
        # Don't cache very large tensors, create them fresh
        codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
        codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
        codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
        return codes_0, codes_1, codes_2
    
    cache_key = num_frames
    if cache_key not in _tensor_buffer_cache:
        # Create and cache new tensors
        codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
        codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
        codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
        _tensor_buffer_cache[cache_key] = (codes_0, codes_1, codes_2)
    
    return _tensor_buffer_cache[cache_key]

def warmup_snac_model():
    """
    Warm up the SNAC model to eliminate cold start penalty.
    
    This function performs a dummy inference to initialize GPU kernels,
    allocate memory, and compile CUDA code. The first real inference 
    will then be much faster (~5ms instead of ~500ms).
    """
    logger.info("🔥 Starting SNAC model warmup...")
    warmup_start = time.perf_counter()
    
    try:
        # Create dummy codes that match the expected input format
        # Use minimal size but valid tensor structure
        dummy_codes_0 = torch.zeros((1, 1), dtype=torch.int32, device=snac_device)
        dummy_codes_1 = torch.zeros((1, 2), dtype=torch.int32, device=snac_device) 
        dummy_codes_2 = torch.zeros((1, 4), dtype=torch.int32, device=snac_device)
        dummy_codes = [dummy_codes_0, dummy_codes_1, dummy_codes_2]
        
        # Perform warmup inference
        with torch.inference_mode():
            _ = model.decode(dummy_codes)
            
        warmup_time = time.perf_counter() - warmup_start
        logger.info(f"✅ SNAC model warmup completed in {warmup_time*1000:.2f}ms")
        
        # Pre-warm tensor cache for common frame counts
        logger.info("🔥 Pre-warming tensor cache...")
        for frames in [1, 4, 7, 8]:  # Common frame counts
            _ = _get_cached_tensors(frames)
        
        # Clean up warmup tensors
        del dummy_codes_0, dummy_codes_1, dummy_codes_2, dummy_codes
        if snac_device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        warmup_time = time.perf_counter() - warmup_start
        logger.warning(f"⚠️  SNAC warmup failed after {warmup_time*1000:.2f}ms: {e}")
        logger.warning("Model will still work but first inference will be slower")

# Automatically warm up the model on import
if os.getenv("ENABLE_SNAC_WARMUP", "true").lower() == "true":
    warmup_snac_model()

# Local cache to avoid repeated parsing of the same token strings
_token_id_cache = {}
_MAX_CACHE_SIZE = 25000
_CUSTOM_TOKEN_PREFIX = "<custom_token_"


def turn_token_into_id(token_string, index):
    """Convert a custom token string to its numeric ID with caching.

    Args:
        token_string (str): The literal token text coming from the model.
        index (int): Absolute token position (used for offset calculation).

    Returns:
        Optional[int]: Numeric token ID or ``None`` if the token is invalid.
    """
    with DecoderProfilerTimer("Token_ID_Conversion"):
        token_string = token_string.strip()
        mod = index % 7  # Offset cycles every 7 tokens
        cache_key = (token_string, mod)

        with DecoderProfilerTimer("Token_Cache_Lookup"):
            if cache_key in _token_id_cache:
                return _token_id_cache[cache_key]

        with DecoderProfilerTimer("Token_String_Parsing"):
            # Locate the last occurrence of the custom token pattern (mirrors original logic)
            last_idx = token_string.rfind(_CUSTOM_TOKEN_PREFIX)
            if last_idx == -1:
                if len(_token_id_cache) < _MAX_CACHE_SIZE:
                    _token_id_cache[cache_key] = None
                return None

            token_substr = token_string[last_idx:]  # from prefix to end

            if not token_substr.startswith(_CUSTOM_TOKEN_PREFIX) or not token_substr.endswith(">"):
                if len(_token_id_cache) < _MAX_CACHE_SIZE:
                    _token_id_cache[cache_key] = None
                return None

            digits = token_substr[len(_CUSTOM_TOKEN_PREFIX):-1]
            if not digits.isdigit():
                if len(_token_id_cache) < _MAX_CACHE_SIZE:
                    _token_id_cache[cache_key] = None
                return None

            token_id = int(digits) - 10 - (mod * 4096)

            if len(_token_id_cache) < _MAX_CACHE_SIZE:
                _token_id_cache[cache_key] = token_id

            return token_id


def convert_to_audio(multiframe, count):
    """
    Highly optimized version of convert_to_audio that eliminates inefficient 
    tensor operations and reduces CPU-GPU transfers for much faster inference
    on high-end GPUs.
    """
    with DecoderProfilerTimer("Audio_Conversion_Total"):
        with DecoderProfilerTimer("Audio_Frame_Validation"):
            if len(multiframe) < 7:
                return None
            
            num_frames = len(multiframe) // 7
        
        with DecoderProfilerTimer("Tensor_Allocation"):
            # Get cached tensors or create new ones - avoids repeated allocation
            codes_0, codes_1, codes_2 = _get_cached_tensors(num_frames)
        
        with DecoderProfilerTimer("Tensor_Population"):
            # Vectorized tensor population - much faster than Python loops
            # Convert multiframe to tensor once for vectorized operations
            multiframe_tensor = torch.tensor(multiframe[:num_frames * 7], dtype=torch.int32, device=snac_device)
            multiframe_view = multiframe_tensor.view(num_frames, 7)
            
            # Use tensor slicing and indexing instead of loops
            codes_0[0, :] = multiframe_view[:, 0]                    # Every 7th element starting at 0
            codes_1[0, 0::2] = multiframe_view[:, 1]                 # Every 7th element starting at 1 
            codes_1[0, 1::2] = multiframe_view[:, 4]                 # Every 7th element starting at 4
            codes_2[0, 0::4] = multiframe_view[:, 2]                 # Every 7th element starting at 2
            codes_2[0, 1::4] = multiframe_view[:, 3]                 # Every 7th element starting at 3  
            codes_2[0, 2::4] = multiframe_view[:, 5]                 # Every 7th element starting at 5
            codes_2[0, 3::4] = multiframe_view[:, 6]                 # Every 7th element starting at 6
        
        with DecoderProfilerTimer("Tensor_Validation"):
            # Optimized validation using tensor operations
            # Check all tensors at once with logical operations
            valid_0 = torch.all((codes_0 >= 0) & (codes_0 <= 4096))
            valid_1 = torch.all((codes_1 >= 0) & (codes_1 <= 4096))
            valid_2 = torch.all((codes_2 >= 0) & (codes_2 <= 4096))
            
            if not (valid_0 and valid_1 and valid_2):
                return None
        
        codes = [codes_0, codes_1, codes_2]
        
        with DecoderProfilerTimer("SNAC_Model_Decode"):
            with torch.inference_mode():   
                audio_hat = model.decode(codes)
        
        with DecoderProfilerTimer("Audio_Post_Processing"):
            audio_slice = audio_hat[:, :, 2048:4096]
            
            if snac_device == "cuda":
                with DecoderProfilerTimer("GPU_to_CPU_Transfer"):
                    audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
                    return audio_int16_tensor.cpu().numpy().tobytes()
            else:
                with DecoderProfilerTimer("CPU_Audio_Processing"):
                    audio_np = audio_slice.numpy()
                    return (audio_np * 32767.0).round().astype(np.int16).tobytes()


async def tokens_decoder(token_gen):
    """Decode tokens into audio chunks with reduced latency.

    The first audio chunk is emitted as soon as **one** frame (7 tokens) is
    available, drastically reducing time-to-first-byte. Subsequent chunks are
    processed every 7 tokens using a sliding window of the last 4 frames (28
    tokens) mirroring the original behaviour.
    """
    with DecoderProfilerTimer("Tokens_Decoder_Total"):
        buffer = []
        count = 0
        first_chunk_sent = False
        MIN_FRAMES_FIRST = 7      # 1 frame for ultra-low latency
        MIN_FRAMES_SUBSEQ = 28    # 4 frames
        PROCESS_EVERY = 7         # process at every full frame boundary
        
        first_audio_start = None
        total_tokens_processed = 0
        total_audio_chunks_generated = 0

        async for token_sim in token_gen:
            with DecoderProfilerTimer("Token_Processing_Loop"):
                with DecoderProfilerTimer("Token_Conversion"):
                    token = turn_token_into_id(token_sim, count)
                    if token is None or token <= 0:
                        continue

                buffer.append(token)
                count += 1
                total_tokens_processed += 1

                if not first_chunk_sent and count >= MIN_FRAMES_FIRST:
                    if first_audio_start is None:
                        first_audio_start = time.perf_counter()
                    
                    with DecoderProfilerTimer("First_Audio_Chunk_Generation"):
                        audio = convert_to_audio(buffer[-MIN_FRAMES_FIRST:], count)
                        if audio is not None:
                            first_chunk_sent = True
                            total_audio_chunks_generated += 1
                            _decoder_profiling_data["First_Audio_Chunk_Size"].append(len(audio))
                            yield audio
                            
                elif first_chunk_sent and count % PROCESS_EVERY == 0:
                    with DecoderProfilerTimer("Subsequent_Audio_Chunk_Generation"):
                        audio = convert_to_audio(buffer[-MIN_FRAMES_SUBSEQ:], count)
                        if audio is not None:
                            total_audio_chunks_generated += 1
                            _decoder_profiling_data["Audio_Chunk_Size"].append(len(audio))
                            yield audio
        
        # Log final statistics
        _decoder_profiling_data["Total_Tokens_Processed"].append(total_tokens_processed)
        _decoder_profiling_data["Total_Audio_Chunks_Generated"].append(total_audio_chunks_generated)
        logger.info(f"🎵 Total_Tokens_Processed: {total_tokens_processed}")
        logger.info(f"🎵 Total_Audio_Chunks_Generated: {total_audio_chunks_generated}")
        if total_tokens_processed > 0:
            efficiency = total_audio_chunks_generated / total_tokens_processed
            _decoder_profiling_data["Audio_Generation_Efficiency"].append(efficiency)
            logger.info(f"🎵 Audio_Generation_Efficiency: {efficiency:.3f}")


def tokens_decoder_sync(syn_token_gen):

    audio_queue = queue.Queue()

    # Convert the synchronous token generator into an async generator.
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        # tokens_decoder.tokens_decoder is assumed to be an async generator that processes tokens.
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()


def get_decoder_profiling_summary():
    """Get a summary of all decoder profiling data collected."""
    if not _decoder_profiling_data:
        return "No decoder profiling data collected."
        
    summary = ["\n=== DECODER PROFILING SUMMARY ==="]
    total_times = defaultdict(float)
    
    for operation, times in _decoder_profiling_data.items():
        if times and isinstance(times[0], (int, float)):
            # Handle timing data
            if "Size" in operation or "Processed" in operation or "Generated" in operation or "Efficiency" in operation:
                # Handle non-timing metrics
                avg_value = sum(times) / len(times)
                min_value = min(times)
                max_value = max(times)
                total_value = sum(times)
                
                summary.append(f"{operation}:")
                summary.append(f"  Count: {len(times)}")
                summary.append(f"  Average: {avg_value:.2f}")
                summary.append(f"  Min: {min_value:.2f}")
                summary.append(f"  Max: {max_value:.2f}")
                summary.append(f"  Total: {total_value:.2f}")
                summary.append("")
            else:
                # Handle timing metrics
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                total_time = sum(times)
                total_times[operation] = total_time
                
                summary.append(f"{operation}:")
                summary.append(f"  Count: {len(times)}")
                summary.append(f"  Average: {avg_time*1000:.2f}ms")
                summary.append(f"  Min: {min_time*1000:.2f}ms")
                summary.append(f"  Max: {max_time*1000:.2f}ms")
                summary.append(f"  Total: {total_time*1000:.2f}ms")
                summary.append("")
    
    # Sort operations by total time to identify bottlenecks
    if total_times:
        summary.append("Top decoder bottlenecks by total time:")
        sorted_ops = sorted(total_times.items(), key=lambda x: x[1], reverse=True)
        for i, (op, total_time) in enumerate(sorted_ops[:5]):
            summary.append(f"  {i+1}. {op}: {total_time*1000:.2f}ms")
    
    summary.append("=================================")
    return "\n".join(summary)


def clear_decoder_profiling_data():
    """Clear all collected decoder profiling data."""
    global _decoder_profiling_data
    _decoder_profiling_data.clear()