"""
Example client for word timestamp WebSocket endpoint.
Shows how to consume audio and timeline updates.
"""

import asyncio
import json
import websockets
import wave
import io
from typing import Dict, List

class TTSClient:
    """Client for TTS with word timestamps."""
    
    def __init__(self, uri: str = "ws://localhost:8000/v1/tts"):
        self.uri = uri
        self.sample_rate = 24000
        self.channels = 1
        
    async def synthesize(self, text: str, voice: str = "tara", 
                         speaking_rate: str = "normal") -> Dict:
        """
        Synthesize text with word timestamps.
        Returns dict with audio bytes and timeline.
        """
        result = {
            "audio_chunks": [],
            "timeline": None,
            "meta": None,
            "error": None
        }
        
        async with websockets.connect(self.uri) as ws:
            # Send request
            request = {
                "text": text,
                "voice_id": voice,
                "speaking_rate": speaking_rate,
                "segment_id": f"segment_{hash(text)}"
            }
            
            await ws.send(json.dumps(request))
            
            # Receive responses
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    
                    if isinstance(message, bytes):
                        # Audio chunk
                        result["audio_chunks"].append(message)
                    else:
                        # JSON message
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "META":
                            result["meta"] = data
                            
                        elif msg_type in ["TIMELINE_UPDATE", "FINAL"]:
                            result["timeline"] = data
                            if msg_type == "FINAL":
                                break
                                
                        elif msg_type == "ERROR":
                            result["error"] = data.get("message")
                            break
                            
                except asyncio.TimeoutError:
                    result["error"] = "Timeout waiting for response"
                    break
        
        return result
    
    def save_audio(self, audio_chunks: List[bytes], filename: str):
        """Save audio chunks to WAV file."""
        # Combine all chunks
        audio_data = b''.join(audio_chunks)
        
        # Create WAV file
        with wave.open(filename, 'wb') as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_data)
        
        print(f"Audio saved to {filename}")
    
    def print_timeline(self, timeline: Dict):
        """Pretty print the timeline."""
        if not timeline:
            print("No timeline available")
            return
            
        words = timeline.get("words", [])
        starts = timeline.get("start", [])
        ends = timeline.get("end", [])
        finalized = timeline.get("finalized_until_index", -1)
        
        print("\nWord Timeline:")
        print("-" * 60)
        for i, word in enumerate(words):
            status = "✓" if i <= finalized else " "
            print(f"{status} {i:2d}. {word:15s} {starts[i]:7.3f}s - {ends[i]:7.3f}s")
        
        if timeline.get("type") == "FINAL":
            print("-" * 60)
            print(f"Total duration: {timeline.get('duration_sec', 0):.3f}s")
            print(f"Total bytes: {timeline.get('total_bytes', 0):,}")


async def main():
    """Example usage of TTS client."""
    client = TTSClient()
    
    # Test texts
    test_texts = [
        "Hello world, this is a test of word timestamps.",
        "The quick brown fox jumps over the lazy dog.",
        "How are you doing today? I hope you're having a great day!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: {text}")
        print('='*60)
        
        # Synthesize with word timestamps
        result = await client.synthesize(text)
        
        if result["error"]:
            print(f"Error: {result['error']}")
            continue
        
        # Print meta info
        if result["meta"]:
            meta = result["meta"]
            print(f"\nAudio format: {meta['format']}, {meta['sample_rate']}Hz, {meta['channels']} channel(s)")
        
        # Print timeline
        client.print_timeline(result["timeline"])
        
        # Save audio
        if result["audio_chunks"]:
            filename = f"output_{i}.wav"
            client.save_audio(result["audio_chunks"], filename)
            print(f"\nReceived {len(result['audio_chunks'])} audio chunks")
        
        # Demonstrate interruption point calculation
        if result["timeline"]:
            timeline = result["timeline"]
            words = timeline.get("words", [])
            ends = timeline.get("end", [])
            
            # Find best cut point at 2 seconds
            cut_time = 2.0
            guard_ms = 60
            cut_point = -1
            
            for j, end_time in enumerate(ends):
                if end_time <= cut_time - (guard_ms / 1000):
                    cut_point = j
            
            if cut_point >= 0:
                print(f"\nInterruption at {cut_time}s would cut after word '{words[cut_point]}' at {ends[cut_point]:.3f}s")


if __name__ == "__main__":
    print("TTS Client Example with Word Timestamps")
    print("Make sure the server is running: uvicorn main:app --host 0.0.0.0 --port 8000")
    print()
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")