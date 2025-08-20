#!/usr/bin/env python3
"""
Download/generate TTS audio from Orpheus TTS server.

Writes a folder structure like:
  outputs/tts_download/orpheus/
    item_00000/
      text.txt
      sample_1.wav
      sample_2.wav
    item_00001/
      ...
"""

import os
import json
import struct
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

# Audio parameters (matching benchmark.py)
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
CHANNELS = 1

DEFAULT_FIXED_TEXT = (
    "I have documented that you intend to pay four hundred thirty-two dollars and sixty-one cents on September seventeenth, twenty twenty-five. This note does not move any funds automatically."
)


def generate_wav_header(data_size: int) -> bytes:
    """Generate WAV header for PCM audio data."""
    bytes_per_sample = BITS_PER_SAMPLE // 8
    block_align = bytes_per_sample * CHANNELS
    byte_rate = SAMPLE_RATE * block_align
    
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
    header.extend(struct.pack('<H', CHANNELS))
    header.extend(struct.pack('<I', SAMPLE_RATE))
    header.extend(struct.pack('<I', byte_rate))
    header.extend(struct.pack('<H', block_align))
    header.extend(struct.pack('<H', BITS_PER_SAMPLE))
    # Data chunk
    header.extend(b'data')
    header.extend(struct.pack('<I', data_size))
    
    return bytes(header)


def synthesize_text(text: str, base_url: str, session: requests.Session) -> Optional[bytes]:
    """Synthesize text using Orpheus TTS server."""
    try:
        response = session.post(
            f"{base_url}/v1/audio/speech/stream",
            json={
                "input": text,
                "voice": "tara"
            },
            stream=True,
            timeout=30
        )
        response.raise_for_status()
        
        # Read PCM data
        audio_data = bytearray()
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                audio_data.extend(chunk)
        
        if not audio_data:
            return None
        
        # Add WAV header
        wav_header = generate_wav_header(len(audio_data))
        return wav_header + audio_data
        
    except Exception as e:
        print(f"Synthesis error: {e}")
        return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def extract_text(item: Dict[str, Any], text_key: Optional[str]) -> Optional[str]:
    if text_key and text_key in item and isinstance(item[text_key], str):
        return item[text_key]
    for k in ["text", "verbalized", "verbalized_text", "prompt"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def main(args):
    # Create session for Orpheus server
    base_url = args.base_url
    
    if args.use_fixed_text:
        num_items = max(1, int(args.num_items))
        items = [{"text": (args.fixed_text or DEFAULT_FIXED_TEXT)} for _ in range(num_items)]
    else:
        items = read_jsonl(Path(args.input))
    if args.limit is not None:
        items = items[: max(0, args.limit)]

    # Fixed output directory
    run_dir = Path(args.output_dir) / "orpheus"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing to: {run_dir}")

    # Pre-create all item folders and text files
    prepared: List[tuple[int, Path, str]] = []
    for idx, it in enumerate(items):
        text = extract_text(it, args.text_key)
        if not text:
            continue
        item_dir = run_dir / f"item_{idx:05d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "text.txt").write_text(text)
        prepared.append((idx, item_dir, text))

    # Create session for connection reuse
    with requests.Session() as session:
        # Process with thread pool
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = []
            
            def produce_one(p_idx: int, p_item_dir: Path, p_text: str, sample_index: int) -> None:
                audio = synthesize_text(p_text, base_url, session)
                if audio:
                    out_path = p_item_dir / f"sample_{sample_index+1}.wav"
                    with open(out_path, "wb") as f:
                        f.write(audio)
            
            for idx, item_dir, text in prepared:
                for s in range(args.samples_per_prompt):
                    future = executor.submit(produce_one, idx, item_dir, text, s)
                    futures.append(future)
            
            total = len(futures)
            print(f"Launching {total} synthesis tasks with concurrency={args.concurrency}...")
            
            # Process with progress bar
            for future in tqdm(as_completed(futures), total=total, desc="Synthesizing"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in synthesis: {e}")

    print(f"Done. Folder ready for evaluation: {run_dir}")


def build_parser():
    import argparse
    p = argparse.ArgumentParser(description="Download TTS audio from Orpheus server")
    p.add_argument("--base-url", default="http://localhost:9090", help="Base URL for TTS server")
    # JSONL mode
    p.add_argument("--input", help="Path to prompts JSONL")
    p.add_argument("--text-key", help="Field name for text in JSONL (auto-detect if omitted)")
    # Fixed text mode
    p.add_argument("--use-fixed-text", action="store_true", help="Use a fixed hardcoded text instead of JSONL")
    p.add_argument("--fixed-text", help="Override the default fixed text")
    p.add_argument("--num-items", type=int, default=500, help="Number of items when using fixed text")
    p.add_argument("--samples-per-prompt", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--output-dir", default="outputs/tts_download")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)


"""
Usage:
python generate_audio.py \
  --use-fixed-text \
  --num-items 10 \
  --samples-per-prompt 2 \
  --concurrency 4
"""