#!/usr/bin/env python3
"""
Standalone test for word timestamps without requiring the model.
Simulates audio generation to test the timing logic.
"""

import asyncio
import json
import time
from src.word_timestamps import WordTimestampGenerator

def test_basic_timeline():
    """Test basic timeline generation."""
    print("="*60)
    print("TEST 1: Basic Timeline Generation")
    print("="*60)
    
    gen = WordTimestampGenerator()
    
    test_cases = [
        ("Hello world!", "tara", "normal"),
        ("The quick brown fox jumps over the lazy dog.", "tara", "normal"),
        ("How are you? I'm fine, thanks!", "tara", "normal"),
        ("One. Two. Three. Four. Five.", "tara", "slow"),
    ]
    
    for text, voice, rate in test_cases:
        print(f"\nText: '{text}'")
        print(f"Voice: {voice}, Rate: {rate}")
        
        timeline = gen.create_timeline(text, voice, rate)
        
        if timeline:
            print(f"  Words: {timeline.words}")
            print(f"  Punct: {timeline.punct_classes}")
            print(f"  Phones: {timeline.phone_counts}")
            
            print("\n  Initial timeline:")
            for i, word in enumerate(timeline.words):
                print(f"    {i:2d}. {word:15s} {timeline.start[i]:7.3f}s - {timeline.end[i]:7.3f}s")
            
            print(f"\n  Total predicted duration: {timeline.end[-1]:.3f}s")


def test_rescaling():
    """Test timeline rescaling at different playback points."""
    print("\n" + "="*60)
    print("TEST 2: Timeline Rescaling")
    print("="*60)
    
    gen = WordTimestampGenerator()
    text = "The quick brown fox jumps over the lazy dog."
    timeline = gen.create_timeline(text, "tara", "normal")
    
    print(f"Text: '{text}'")
    print(f"Initial duration: {timeline.end[-1]:.3f}s")
    
    # Simulate playback at different points
    playback_points = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    for audio_time in playback_points:
        print(f"\n--- Audio at {audio_time:.1f}s ---")
        
        # Reset to base timeline
        timeline.start = timeline.base_start.copy()
        timeline.end = timeline.base_end.copy()
        timeline.finalized_index = -1
        
        # Rescale
        updated = timeline.rescale_to_audio_time(audio_time)
        
        print(f"  Rescaling {'applied' if updated else 'not needed'}")
        print(f"  Finalized up to word {timeline.finalized_index}")
        
        # Show first few words
        for i in range(min(5, len(timeline.words))):
            status = "✓" if i <= timeline.finalized_index else "○"
            print(f"    {status} {timeline.words[i]:10s} {timeline.start[i]:6.3f}s - {timeline.end[i]:6.3f}s")


async def test_streaming_simulation():
    """Simulate streaming with progressive timeline updates."""
    print("\n" + "="*60)
    print("TEST 3: Streaming Simulation")
    print("="*60)
    
    gen = WordTimestampGenerator()
    text = "Hello world, this is a streaming test with word timestamps."
    timeline = gen.create_timeline(text, "tara", "normal")
    
    print(f"Text: '{text}'")
    print(f"Predicted duration: {timeline.end[-1]:.3f}s")
    
    # Simulate audio generation at 24kHz, 16-bit mono
    SAMPLE_RATE = 24000
    BYTES_PER_SAMPLE = 2
    BYTES_PER_SEC = SAMPLE_RATE * BYTES_PER_SAMPLE
    
    # Simulate chunks of 100ms each
    CHUNK_MS = 100
    CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_MS / 1000)
    
    # Simulate faster/slower synthesis (1.0 = realtime)
    SYNTHESIS_SPEED = 0.8  # Slightly slower than predicted
    
    total_duration = timeline.end[-1] * SYNTHESIS_SPEED
    total_bytes = int(total_duration * BYTES_PER_SEC)
    num_chunks = (total_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES
    
    print(f"\nSimulating {num_chunks} chunks of {CHUNK_MS}ms each")
    print(f"Actual synthesis speed: {SYNTHESIS_SPEED}x predicted\n")
    
    bytes_sent = 0
    last_update_time = 0
    update_interval = timeline.voice_profile.rescale_interval_ms / 1000
    
    for chunk_idx in range(num_chunks):
        # Simulate sending a chunk
        bytes_sent += CHUNK_BYTES
        audio_time = bytes_sent / BYTES_PER_SEC
        
        # Check if we should rescale
        if audio_time - last_update_time >= update_interval:
            if timeline.rescale_to_audio_time(audio_time):
                print(f"Chunk {chunk_idx:3d} | Audio: {audio_time:5.2f}s | Finalized: {timeline.finalized_index:2d}")
                
                # Show current state of first few words
                for i in range(min(3, len(timeline.words))):
                    status = "✓" if i <= timeline.finalized_index else "○"
                    print(f"  {status} {timeline.words[i]:8s} {timeline.start[i]:5.3f}s - {timeline.end[i]:5.3f}s")
                
                last_update_time = audio_time
        
        # Simulate processing time
        await asyncio.sleep(0.001)
    
    # Final state
    print(f"\nFinal state after {bytes_sent/BYTES_PER_SEC:.3f}s of audio:")
    timeline.finalized_index = len(timeline.words) - 1
    for i, word in enumerate(timeline.words):
        print(f"  ✓ {word:15s} {timeline.start[i]:6.3f}s - {timeline.end[i]:6.3f}s")


def test_interruption_semantics():
    """Test finding the best interruption point."""
    print("\n" + "="*60)
    print("TEST 4: Interruption Semantics")
    print("="*60)
    
    gen = WordTimestampGenerator()
    text = "The quick brown fox jumps over the lazy dog."
    timeline = gen.create_timeline(text, "tara", "normal")
    
    print(f"Text: '{text}'")
    print(f"Guard time: {timeline.voice_profile.guard_ms}ms\n")
    
    # Test interruption at different times
    interrupt_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for interrupt_time in interrupt_times:
        # Find best cut point
        guard_sec = timeline.voice_profile.guard_ms / 1000
        cut_index = -1
        
        for i, end_time in enumerate(timeline.end):
            if end_time <= interrupt_time - guard_sec:
                cut_index = i
        
        if cut_index >= 0:
            cut_word = timeline.words[cut_index]
            cut_time = timeline.end[cut_index]
            cut_sample = int(cut_time * 24000)
            print(f"Interrupt at {interrupt_time:.1f}s → Cut after '{cut_word}' at {cut_time:.3f}s (sample {cut_sample})")
        else:
            print(f"Interrupt at {interrupt_time:.1f}s → No safe cut point (too early)")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 5: Edge Cases")
    print("="*60)
    
    gen = WordTimestampGenerator()
    
    edge_cases = [
        "",  # Empty text
        "   ",  # Whitespace only
        "Word",  # Single word
        "...",  # Only punctuation
        "Hello...",  # Trailing punctuation
        "?!",  # Only punctuation marks
        "A a I i",  # Very short words
        "Supercalifragilisticexpialidocious",  # Very long word
    ]
    
    for text in edge_cases:
        print(f"\nText: '{text}'")
        timeline = gen.create_timeline(text, "tara", "normal")
        
        if timeline:
            print(f"  Success: {len(timeline.words)} words")
            for i, word in enumerate(timeline.words):
                print(f"    {word:30s} {timeline.start[i]:6.3f}s - {timeline.end[i]:6.3f}s")
        else:
            print("  No timeline (empty or invalid)")


if __name__ == "__main__":
    print("STANDALONE WORD TIMESTAMP TESTS")
    print("="*60)
    print("Testing without model or server\n")
    
    # Run all tests
    test_basic_timeline()
    test_rescaling()
    asyncio.run(test_streaming_simulation())
    test_interruption_semantics()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)