#!/usr/bin/env python3
"""
Test client for word timestamps WebSocket endpoint.
Tests the actual server with various text normalization cases.
"""

import asyncio
import json
import websockets
import sys
from typing import List, Dict

async def test_single_text(uri: str, text: str, voice_id: str = "default"):
    """Test a single text input and collect timeline updates."""
    print(f"\nTesting: \"{text}\"")
    print("-" * 60)
    
    async with websockets.connect(uri) as websocket:
        # Send request
        request = {
            "text": text,
            "voice_id": voice_id,
            "speaking_rate": "normal"
        }
        
        await websocket.send(json.dumps(request))
        
        # Collect responses
        timeline_updates = []
        audio_bytes = 0
        meta_received = False
        final_received = False
        
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                
                if isinstance(message, bytes):
                    # Audio chunk
                    audio_bytes += len(message)
                    # Print progress every 48KB (1 second of audio)
                    if audio_bytes % 48000 < len(message):
                        print(f"  Audio received: {audio_bytes:,} bytes ({audio_bytes/48000:.1f}s)")
                else:
                    # JSON message
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "META":
                        meta_received = True
                        print(f"  META: sample_rate={data.get('sample_rate')}, format={data.get('format')}")
                    
                    elif msg_type == "TIMELINE_UPDATE":
                        timeline_updates.append(data)
                        words = data.get('words', [])
                        finalized = data.get('finalized_until_index', -1)
                        
                        # Show timeline update summary
                        if len(timeline_updates) == 1:
                            print(f"  Initial timeline: {len(words)} words")
                            # Show first few words with timing
                            for i in range(min(3, len(words))):
                                start = data['start'][i]
                                end = data['end'][i]
                                dur_ms = (end - start) * 1000
                                print(f"    '{words[i]}': {start:.3f}s - {end:.3f}s ({dur_ms:.0f}ms)")
                        else:
                            finalized_word = words[finalized] if finalized >= 0 else "none"
                            print(f"  Update #{len(timeline_updates)}: finalized through '{finalized_word}' (index {finalized})")
                    
                    elif msg_type == "FINAL":
                        final_received = True
                        duration = data.get('duration_sec', 0)
                        total_bytes = data.get('total_bytes', 0)
                        print(f"  FINAL: duration={duration:.3f}s, bytes={total_bytes:,}")
                        break
                    
                    elif msg_type == "ERROR":
                        print(f"  ERROR: {data.get('message')}")
                        break
                        
            except asyncio.TimeoutError:
                print("  Timeout waiting for response")
                break
        
        # Summary
        print("\nSummary:")
        print(f"  Timeline updates: {len(timeline_updates)}")
        print(f"  Total audio bytes: {audio_bytes:,}")
        
        if timeline_updates:
            final_timeline = timeline_updates[-1]
            words = final_timeline.get('words', [])
            if words:
                predicted_duration = final_timeline['end'][-1]
                print(f"  Words processed: {len(words)}")
                print(f"  Predicted duration: {predicted_duration:.3f}s")
                
                # Check if timestamps include sample_rate
                if 'sample_rate' in final_timeline:
                    print(f"  Sample rate included: {final_timeline['sample_rate']} Hz")
                
                # Show interesting words with expanded phonemes
                interesting_words = []
                for i, word in enumerate(words):
                    dur_ms = (final_timeline['end'][i] - final_timeline['start'][i]) * 1000
                    # Words over 1 second are likely expanded numbers/currency
                    if dur_ms > 1000:
                        interesting_words.append((word, dur_ms))
                
                if interesting_words:
                    print(f"\n  Expanded words (likely numbers/currency):")
                    for word, dur_ms in interesting_words:
                        print(f"    '{word}': {dur_ms:.0f}ms")


async def run_tests():
    """Run comprehensive tests against the server."""
    uri = "ws://localhost:8000/v1/tts"
    
    print("=" * 60)
    print("WORD TIMESTAMPS SERVER TEST")
    print("=" * 60)
    
    # Test cases focusing on text normalization
    test_cases = [
        # Basic
        "Hello world.",
        
        # Currency expansion
        "Pay $9.99 today, or $12.50 tomorrow.",
        
        # Phone number
        "Call 1-800-555-1212.",
        
        # Times and dates
        "Meeting at 10:30am on Dec 25th.",
        
        # Numbers and units
        "Speed is 3.14 m/s.",
        
        # Multiple sentences with pauses
        "First sentence. Second one! Third? Fourth.",
        
        # Complex mixed content
        "Order #1234 costs $99.99 - call 555-0123 by 3pm.",
    ]
    
    for text in test_cases:
        try:
            await test_single_text(uri, text)
        except Exception as e:
            print(f"  Error: {e}")
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        print("Connecting to server at ws://localhost:8000/v1/tts")
        print("Make sure server is running with:")
        print("  conda activate trt_clean")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
        print()
        
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nFailed to connect: {e}")
        print("Is the server running?")
        sys.exit(1)