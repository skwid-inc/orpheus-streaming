"""
Test script for word timestamp functionality.
"""

import asyncio
import json
import websockets
import wave
import io
from typing import List, Dict

async def test_word_timestamps():
    """Test the word timestamp WebSocket endpoint."""
    
    uri = "ws://localhost:8000/v1/tts"
    
    test_texts = [
        "Hello world, this is a test.",
        "How are you doing today? I hope you're well!",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Testing: {text}")
        print('='*60)
        
        async with websockets.connect(uri) as websocket:
            # Send request
            request = {
                "text": text,
                "voice_id": "tara",
                "speaking_rate": "normal",
                "segment_id": f"test_{hash(text)}"
            }
            
            await websocket.send(json.dumps(request))
            
            # Collect responses
            timeline_updates = []
            audio_chunks = []
            meta_received = False
            
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    
                    if isinstance(message, bytes):
                        # Audio chunk
                        audio_chunks.append(message)
                        print(f"  Received audio chunk: {len(message)} bytes")
                    else:
                        # JSON message
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "META":
                            meta_received = True
                            print(f"  META: sr={data['sample_rate']}, format={data['format']}")
                        
                        elif msg_type == "TIMELINE_UPDATE":
                            timeline_updates.append(data)
                            print(f"  Timeline update #{len(timeline_updates)}:")
                            print(f"    Words: {data['words']}")
                            print(f"    Finalized: {data['finalized_until_index']}")
                            
                            # Show first 3 word timings
                            for i in range(min(3, len(data['words']))):
                                print(f"    '{data['words'][i]}': {data['start'][i]:.3f}s - {data['end'][i]:.3f}s")
                        
                        elif msg_type == "FINAL":
                            print(f"\n  FINAL event received:")
                            print(f"    Duration: {data['duration_sec']:.3f}s")
                            print(f"    Total bytes: {data['total_bytes']}")
                            print(f"    All words finalized: {data['finalized_until_index'] == len(data['words']) - 1}")
                            break
                        
                        elif msg_type == "ERROR":
                            print(f"  ERROR: {data.get('message')}")
                            break
                            
                except asyncio.TimeoutError:
                    print("  Timeout waiting for response")
                    break
            
            # Summary
            print(f"\nSummary:")
            print(f"  Timeline updates received: {len(timeline_updates)}")
            print(f"  Audio chunks received: {len(audio_chunks)}")
            print(f"  Total audio bytes: {sum(len(c) for c in audio_chunks)}")
            
            if timeline_updates:
                final = timeline_updates[-1]
                print(f"  Words processed: {len(final['words'])}")
                print(f"  Total predicted duration: {final['end'][-1]:.3f}s")

def test_local_timeline():
    """Test timeline generation locally without WebSocket."""
    from src.word_timestamps import WordTimestampGenerator
    
    print("\n" + "="*60)
    print("Testing local timeline generation")
    print("="*60)
    
    gen = WordTimestampGenerator()
    
    test_cases = [
        ("Hello world!", "tara", "normal"),
        ("The quick, brown fox jumps. Over the lazy dog?", "tara", "normal"),
        ("One. Two. Three.", "tara", "slow"),
    ]
    
    for text, voice, rate in test_cases:
        print(f"\nText: '{text}'")
        print(f"Voice: {voice}, Rate: {rate}")
        
        timeline = gen.create_timeline(text, voice, rate)
        
        if timeline:
            print(f"  Words: {timeline.words}")
            print(f"  Punctuation: {timeline.punct_classes}")
            print(f"  Phone counts: {timeline.phone_counts}")
            print("\n  Initial timeline:")
            for i, word in enumerate(timeline.words):
                print(f"    {word:15} {timeline.start[i]:6.3f}s - {timeline.end[i]:6.3f}s")
            
            # Simulate rescaling at 1.5 seconds
            print("\n  After rescaling to 1.5s audio:")
            timeline.rescale_to_audio_time(1.5)
            for i, word in enumerate(timeline.words):
                status = "✓" if i <= timeline.finalized_index else " "
                print(f"  {status} {word:15} {timeline.start[i]:6.3f}s - {timeline.end[i]:6.3f}s")

if __name__ == "__main__":
    print("Word Timestamp Test Suite")
    print("=" * 60)
    
    # Test local timeline generation first
    test_local_timeline()
    
    # Then test WebSocket if server is running
    try:
        print("\n" + "="*60)
        print("Testing WebSocket endpoint (make sure server is running)")
        print("="*60)
        asyncio.run(test_word_timestamps())
    except Exception as e:
        print(f"\nWebSocket test failed: {e}")
        print("Make sure the server is running with: uvicorn main:app --host 0.0.0.0 --port 8000")