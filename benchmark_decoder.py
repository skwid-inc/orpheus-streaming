import asyncio
import time
import os
import argparse
from typing import AsyncGenerator, List, Tuple

# Ensure we can import from the local 'src' package
from src.decoder import tokens_decoder


def make_custom_token_string(absolute_index: int, logical_token_id: int) -> str:
    """Create a valid custom token string that the decoder can parse.

    The decoder expects tokens like "<custom_token_XXXX>" where XXXX encodes
    both the logical token id (0..4096) and the absolute position mod 7.

    We also avoid id 0 because the decoder filters out token <= 0.
    """
    logical_token_id = max(1, min(4096, logical_token_id))
    mod = absolute_index % 7
    encoded_value = logical_token_id + 10 + (mod * 4096)
    return f"<custom_token_{encoded_value}>"


async def generate_dummy_tokens(num_frames: int, delay_per_token_s: float = 0.0) -> AsyncGenerator[str, None]:
    """Async generator that yields a stream of valid dummy token strings.

    - num_frames: number of 7-token frames to emit
    - delay_per_token_s: optional delay to simulate upstream token latency
    """
    total_tokens = num_frames * 7
    for i in range(total_tokens):
        # Simple deterministic id pattern in 1..4096
        logical_id = (i % 4096) + 1
        yield make_custom_token_string(i, logical_id)
        if delay_per_token_s > 0:
            await asyncio.sleep(delay_per_token_s)


async def run_decoder_benchmark(num_frames: int, delay_per_token_s: float) -> Tuple[float, float, int, List[int]]:
    """Run the async decoder with dummy tokens and collect metrics.

    Returns a tuple: (ttfb_seconds, total_seconds, total_bytes, chunk_sizes)
    """
    start_time = time.time()
    ttfb = None
    total_bytes = 0
    chunk_sizes: List[int] = []

    async for audio_chunk in tokens_decoder(generate_dummy_tokens(num_frames, delay_per_token_s)):
        if not audio_chunk:
            continue
        if ttfb is None:
            ttfb = time.time() - start_time
        size = len(audio_chunk)
        chunk_sizes.append(size)
        total_bytes += size

    total_time = time.time() - start_time
    return (ttfb or 0.0, total_time, total_bytes, chunk_sizes)


def save_audio_bytes_to_wav(path: str, raw_pcm_bytes: bytes, sample_rate: int = 24000, bits_per_sample: int = 16, channels: int = 1) -> None:
    import struct

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    bytes_per_sample = bits_per_sample // 8
    block_align = bytes_per_sample * channels
    byte_rate = sample_rate * block_align
    data_size = len(raw_pcm_bytes)
    file_size = 36 + data_size

    header = bytearray()
    header.extend(b"RIFF")
    header.extend(struct.pack("<I", file_size))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend(struct.pack("<I", 16))
    header.extend(struct.pack("<H", 1))
    header.extend(struct.pack("<H", channels))
    header.extend(struct.pack("<I", sample_rate))
    header.extend(struct.pack("<I", byte_rate))
    header.extend(struct.pack("<H", block_align))
    header.extend(struct.pack("<H", bits_per_sample))
    header.extend(b"data")
    header.extend(struct.pack("<I", data_size))

    with open(path, "wb") as f:
        f.write(header)
        f.write(raw_pcm_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tokens_decoder with dummy token stream")
    parser.add_argument("--frames", type=int, default=64, help="Number of 7-token frames to emit")
    parser.add_argument("--delay_ms", type=float, default=0.0, help="Optional delay per token in milliseconds")
    parser.add_argument("--save", type=str, default="", help="Optional path to save concatenated audio as WAV")
    args = parser.parse_args()

    delay_per_token_s = max(0.0, args.delay_ms / 1000.0)

    print("Decoder perf test")
    print(f"Frames: {args.frames} (tokens: {args.frames * 7})")
    print(f"Token delay: {delay_per_token_s:.6f}s")

    ttfb, total, total_bytes, chunk_sizes = asyncio.run(run_decoder_benchmark(args.frames, delay_per_token_s))

    print("\nResults:")
    print(f"- TTFB: {ttfb:.4f}s")
    print(f"- Total: {total:.4f}s")
    print(f"- Chunks: {len(chunk_sizes)}")
    if chunk_sizes:
        print(f"  - First chunk bytes: {chunk_sizes[0]}")
        print(f"  - Avg chunk bytes: {sum(chunk_sizes)/len(chunk_sizes):.1f}")
    print(f"- Total audio bytes: {total_bytes}")

    if args.save:
        # Re-run to collect actual bytes for saving, to avoid storing all during timing
        async def gather_bytes():
            data = bytearray()
            async for chunk in tokens_decoder(generate_dummy_tokens(args.frames, delay_per_token_s)):
                if chunk:
                    data.extend(chunk)
            return bytes(data)

        audio_bytes = asyncio.run(gather_bytes())
        save_path = args.save
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "decoder_dummy.wav")
        save_audio_bytes_to_wav(save_path, audio_bytes)
        print(f"Saved audio to: {save_path}")


if __name__ == "__main__":
    main()

import asyncio
import time
import random
import argparse
import statistics

from typing import AsyncIterator, List

from src import decoder as dec


def build_tokens(num_frames: int, seed: int = 1234) -> List[str]:
    rnd = random.Random(seed)
    tokens: List[str] = []
    for i in range(num_frames * 7):
        mod = i % 7
        token_id = rnd.randint(1, 4095)
        raw_val = token_id + 10 + (mod * 4096)
        tokens.append(f"<custom_token_{raw_val}>")
    return tokens


async def synthetic_token_gen_cumulative(tokens: List[str], sleep_ms: int = 0) -> AsyncIterator[str]:
    buf: List[str] = []
    for t in tokens:
        buf.append(t)
        if sleep_ms > 0:
            await asyncio.sleep(sleep_ms / 1000.0)
        yield " ".join(buf)


async def run_once(num_frames: int, sleep_ms: int) -> dict:
    tokens = build_tokens(num_frames)

    orig_convert = dec.convert_to_audio
    call_durations_ms: List[float] = []
    frames_per_call: List[int] = []

    async def wrapped_convert(frame_ids: list[int]):
        if frame_ids is None:
            return None
        start = time.perf_counter()
        try:
            return await orig_convert(frame_ids)
        finally:
            end = time.perf_counter()
            call_durations_ms.append((end - start) * 1000.0)
            frames_per_call.append(len(frame_ids) // 7)

    dec.convert_to_audio = wrapped_convert  # type: ignore

    gen = synthetic_token_gen_cumulative(tokens, sleep_ms=sleep_ms)

    t_start = time.time()
    ttfb = None
    total_bytes = 0

    async for chunk in dec.tokens_decoder(gen):
        if chunk:
            if ttfb is None:
                ttfb = time.time() - t_start
            total_bytes += len(chunk)

    total_time = time.time() - t_start

    # restore
    dec.convert_to_audio = orig_convert  # type: ignore

    bytes_per_second = 24000 * 1 * (16 // 8)
    audio_duration = total_bytes / bytes_per_second if bytes_per_second > 0 else 0.0

    return {
        "ttfb": ttfb or 0.0,
        "total_time": total_time,
        "total_bytes": total_bytes,
        "audio_duration": audio_duration,
        "num_calls": len(call_durations_ms),
        "ms_per_call_mean": statistics.mean(call_durations_ms) if call_durations_ms else 0.0,
        "ms_per_call_p95": statistics.quantiles(call_durations_ms, n=20)[-1] if len(call_durations_ms) >= 20 else (max(call_durations_ms) if call_durations_ms else 0.0),
        "frames_per_call_mean": statistics.mean(frames_per_call) if frames_per_call else 0.0,
        "ms_per_frame_mean": (statistics.mean(call_durations_ms) / (statistics.mean(frames_per_call) or 1)) if call_durations_ms else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark SNAC decoder only (synthetic tokens)")
    parser.add_argument("--frames", type=int, default=120, help="Total frames to generate (7 tokens/frame)")
    parser.add_argument("--sleep_ms", type=int, default=0, help="Delay between tokens to simulate LLM speed")
    args = parser.parse_args()

    res = asyncio.run(run_once(args.frames, args.sleep_ms))

    print("Decoder benchmark (synthetic)")
    print(f"Frames: {args.frames}, Sleep per token: {args.sleep_ms} ms")
    print(f"TTFB: {res['ttfb']:.3f}s, Total: {res['total_time']:.3f}s")
    print(f"Bytes: {res['total_bytes']}, Audio dur: {res['audio_duration']:.3f}s")
    print(f"Calls: {res['num_calls']}, ms/call mean: {res['ms_per_call_mean']:.2f}, p95: {res['ms_per_call_p95']:.2f}")
    print(f"Frames/call mean: {res['frames_per_call_mean']:.2f}, ms/frame mean: {res['ms_per_frame_mean']:.2f}")


if __name__ == "__main__":
    main()


