#!/usr/bin/env python3
"""
Evaluate all WAVs in a folder tree using Gemini, pairing each WAV with a
text.txt next to it. Prints and saves aggregated hallucination results.

Folder structure expected (from generate_audio.py):
  run_dir/
    item_00000/
      text.txt
      sample_1.wav
      sample_2.wav
    item_00001/
      ...

Env (.env):
  GEMINI_API_KEY or GOOGLE_AI_API_KEY
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import asyncio

from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
import shutil

# Import from local files
from async_gemini import evaluate_with_gemini_sync, EVALUATION_PROMPT


async def evaluate_one(model, audio_path: Path, text: str, use_halluc_only: bool, executor: ThreadPoolExecutor) -> Dict[str, Any]:
    audio_bytes = audio_path.read_bytes()
    loop = asyncio.get_event_loop()
    # Always use evaluation prompt
    prompt = EVALUATION_PROMPT
    result = await loop.run_in_executor(
        executor, evaluate_with_gemini_sync, model, audio_bytes, text, prompt
    )
    return result


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv("../.env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in ../.env file")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    # If no input dir provided, default to fixed provider folder used by downloader
    run_dir = Path(args.input_dir) if args.input_dir else Path("outputs/tts_download") / args.provider
    if not run_dir.exists():
        raise ValueError(f"Input dir not found: {run_dir}")

    totals = {"none": 0, "medium": 0, "high": 0}
    details: List[Dict[str, Any]] = []

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    executor = ThreadPoolExecutor(max_workers=max(1, args.workers or args.concurrency))

    eval_targets: List[tuple[Path, str, str]] = []  # (wav_path, text, item_name)
    for item_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        text_path = item_dir / "text.txt"
        if not text_path.exists():
            continue
        text = text_path.read_text()
        for wav in sorted(item_dir.glob("*.wav")):
            eval_targets.append((wav, text, item_dir.name))

    async def run_eval(wav_path: Path, text: str, item_name: str):
        async with semaphore:
            attempts = max(1, args.eval_retries + 1)
            last_err: Exception | None = None
            for _ in range(attempts):
                try:
                    result = await evaluate_one(model, wav_path, text, False, executor)
                    break
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(0.5)
            else:
                print(f"Eval error for {wav_path}: {last_err}")
                totals["high"] += 1
                details.append({
                    "item": item_name,
                    "file": str(wav_path.relative_to(run_dir)),
                    "hallucinations_level": "high",
                    "error": str(last_err),
                })
                return

            level = str(result.get("hallucinations_level", "")).lower()
            if level not in totals:
                level = "high"
            totals[level] += 1

            # Only store details for non-"none" hallucinations to keep JSONL concise
            if level != "none":
                rel = wav_path.relative_to(run_dir)
                dest = samples_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(wav_path, dest)
                except Exception:
                    pass
                details.append({
                    "item": item_name,
                    "file": str(rel),
                    "hallucinations_level": level,
                    "scores": result,
                })

    print(f"Launching {len(eval_targets)} Gemini evals with concurrency={args.concurrency}...")
    tasks = [asyncio.create_task(run_eval(w, t, n)) for (w, t, n) in eval_targets]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        await coro

    total_samples = sum(totals.values())
    print("\n=== Aggregated Hallucinations ===")
    for k in ["none", "medium", "high"]:
        v = totals[k]
        pct = (v / total_samples * 100) if total_samples else 0
        print(f"{k}: {v} ({pct:.1f}%)")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"totals": totals, "total_samples": total_samples}, f, indent=2)
    with open(out_dir / "details.jsonl", "w") as f:
        for row in details:
            f.write(json.dumps(row) + "\n")

    print(f"Saved results to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate WAV folder with Gemini")
    p.add_argument("--provider", choices=["elevenlabs", "cartesia", "baseten"], help="Provider folder to auto-pick when input-dir not given")
    p.add_argument("--input-dir", help="Folder with item_* subdirs and text.txt (defaults to outputs/tts_download/<provider>)")
    p.add_argument("--output-dir", default="outputs/hallucinations_eval")
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--workers", type=int, help="Thread pool size for sync Gemini calls (defaults to --concurrency)")
    # Removed hallucinations-only option - always use evaluation prompt
    p.add_argument("--eval-retries", type=int, default=1, help="Number of retries on Gemini eval failure")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()


"""
Usage:
python eval_with_gemini.py \
  --input-dir outputs/tts_download/orpheus \
  --concurrency 100
"""