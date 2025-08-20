#!/usr/bin/env python3
"""
Comprehensive test suite for word timestamps with text normalization.
Tests various edge cases including numbers, currency, dates, times, phone numbers, etc.
"""

import sys
import json
sys.path.insert(0, 'src')

from word_timestamps import WordTimestampGenerator

def print_timeline_details(text, timeline, show_rescaling=True):
    """Pretty print timeline details for a given text."""
    print(f"\nText: \"{text}\"")
    print("-" * 80)
    
    if not timeline:
        print("No timeline created (empty text)")
        return
    
    # Summary
    print(f"Total words: {len(timeline.words)}")
    print(f"Sentence boundaries: {timeline.sentence_end_indices}")
    print(f"Total duration: {timeline.end[-1]:.3f} seconds")
    
    # Word-by-word breakdown
    print("\n{:<20} {:>8} {:<10} {:>10} {:>10} {:>10}".format(
        "Word", "Phonemes", "Punct", "Start(s)", "End(s)", "Dur(ms)"
    ))
    print("-" * 78)
    
    for i, (word, phones, punct, start, end) in enumerate(zip(
        timeline.words, 
        timeline.phone_counts,
        timeline.punct_classes,
        timeline.start,
        timeline.end
    )):
        dur_ms = (end - start) * 1000
        # Highlight sentence boundaries
        marker = " *" if i in timeline.sentence_end_indices else ""
        print(f"{word:<20} {phones:>8} {punct:<10} {start:>10.3f} {end:>10.3f} {dur_ms:>10.1f}{marker}")
    
    print("\n* = sentence boundary")
    
    # Simulate rescaling if requested
    if show_rescaling and timeline.end[-1] > 0:
        print("\nRescaling simulation (audio playback progress):")
        test_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for audio_time in test_points:
            if audio_time > timeline.end[-1]:
                break
            # Create fresh timeline for testing
            test_timeline = generator.create_timeline(text,  speaking_rate="normal")
            updated = test_timeline.rescale_to_audio_time(audio_time)
            finalized = test_timeline.finalized_index
            finalized_word = test_timeline.words[finalized] if finalized >= 0 else "none"
            print(f"  At {audio_time:.1f}s: finalized through '{finalized_word}' (index {finalized}), rescaled={updated}")


def test_basic_sentences():
    """Test basic sentence structures."""
    print("\n" + "=" * 80)
    print("BASIC SENTENCES")
    print("=" * 80)
    
    test_cases = [
        "Hello world.",
        "This is a test.",
        "How are you today?",
        "Stop! Don't move.",
        "Yes, I agree.",
    ]
    
    for text in test_cases:
        timeline = generator.create_timeline(text,  speaking_rate="normal")
        print_timeline_details(text, timeline, show_rescaling=False)


def test_text_normalization():
    """Test various text normalization cases."""
    print("\n" + "=" * 80)
    print("TEXT NORMALIZATION CASES")
    print("=" * 80)
    
    test_cases = [
        # Currency
        "Pay $9.99 today, or $12.50 tomorrow.",
        "The price is €100 or £75.",
        
        # Times
        "ETA 10:30am? Maybe 11:05am.",
        "Meeting at 3:00pm sharp.",
        
        # Numbers and decimals
        "Speed is 3.14 m/s; peak 12.7 m/s.",
        "Temperature: -5°C to 32°F.",
        
        # Distances and units
        "He ran 5km, then 10km.",
        "The box weighs 2.5kg.",
        
        # Phone numbers
        "Call 1-800-555-1212.",
        "My number is (555) 123-4567.",
        
        # Dates and years
        "In 2025, growth was 12%.",
        "Born on 12/25/1990.",
        
        # Ordinals
        "She finished 1st, he was 2nd.",
        "This is the 21st century.",
        
        # Mixed
        "Order #1234 costs $99.99 - call 555-0123.",
        "Q3 2024 revenue: $1.2M (up 15%).",
    ]
    
    for text in test_cases:
        timeline = generator.create_timeline(text,  speaking_rate="normal")
        print_timeline_details(text, timeline, show_rescaling=True)


def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\n" + "=" * 80)
    print("EDGE CASES")
    print("=" * 80)
    
    test_cases = [
        # Empty and single word
        "",
        "Hello",
        "123",
        "$5",
        
        # No punctuation
        "this is a test without any punctuation at all",
        
        # Multiple sentences
        "First sentence. Second one! Third? Fourth.",
        
        # Long numbers
        "$1,234,567.89",
        "1-800-FLOWERS",
        
        # Special characters
        "user@example.com",
        "C++ programming",
        "50% off!",
        
        # Contractions
        "I'm sure you're right, but we've got to go.",
        "Don't worry, it'll be fine.",
    ]
    
    for text in test_cases:
        timeline = generator.create_timeline(text,  speaking_rate="normal")
        print_timeline_details(text, timeline, show_rescaling=False)


def test_speaking_rates():
    """Test different speaking rates."""
    print("\n" + "=" * 80)
    print("SPEAKING RATE VARIATIONS")
    print("=" * 80)
    
    test_text = "The quick brown fox jumps over the lazy dog."
    
    for rate in ["slow", "normal", "fast"]:
        timeline = generator.create_timeline(test_text,  speaking_rate=rate)
        if timeline:
            duration = timeline.end[-1]
            print(f"\nRate: {rate:6} | Duration: {duration:.3f}s | Text: \"{test_text}\"")


def test_rescaling_behavior():
    """Test rescaling behavior in detail."""
    print("\n" + "=" * 80)
    print("RESCALING BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    # Multi-sentence text to test sentence boundary rescaling
    test_text = "First sentence here. Second sentence now! Third one? Final statement."
    timeline = generator.create_timeline(test_text,  speaking_rate="normal")
    
    if not timeline:
        print("Failed to create timeline")
        return
    
    print(f"\nText: \"{test_text}\"")
    print(f"Words: {timeline.words}")
    print(f"Sentence boundaries at indices: {timeline.sentence_end_indices}")
    print(f"\nOriginal timeline:")
    print("Word index | Word            | Original End | Scaled End | Status")
    print("-" * 70)
    
    # Test rescaling at various points
    audio_times = [0.3, 0.6, 1.0, 1.5, 2.0]
    
    for audio_time in audio_times:
        if audio_time > timeline.end[-1]:
            break
            
        # Create fresh timeline
        test_timeline = generator.create_timeline(test_text,  speaking_rate="normal")
        test_timeline.rescale_to_audio_time(audio_time)
        
        print(f"\nAfter rescaling at {audio_time:.1f}s (finalized up to index {test_timeline.finalized_index}):")
        for i, word in enumerate(test_timeline.words):
            status = "FINALIZED" if i <= test_timeline.finalized_index else "open"
            orig_end = test_timeline.base_end[i]
            scaled_end = test_timeline.end[i]
            diff = scaled_end - orig_end
            print(f"{i:10} | {word:15} | {orig_end:12.3f} | {scaled_end:10.3f} | {status:9} | diff: {diff:+.3f}")


def test_timeline_events():
    """Test timeline event generation."""
    print("\n" + "=" * 80)
    print("TIMELINE EVENT FORMAT")
    print("=" * 80)
    
    test_text = "Hello world, this is a test."
    timeline = generator.create_timeline(test_text,  speaking_rate="normal")
    
    if timeline:
        event = timeline.get_timeline_event()
        print(f"\nText: \"{test_text}\"")
        print("\nGenerated timeline event:")
        print(json.dumps(event, indent=2))
        
        # Verify sample calculations
        print("\nSample calculations verification:")
        sr = event["sample_rate"]
        for i, (word, start, end, start_samp, end_samp) in enumerate(zip(
            event["words"][:3],  # Just show first 3 words
            event["start"][:3],
            event["end"][:3],
            event["start_sample"][:3],
            event["end_sample"][:3]
        )):
            calc_start = int(round(start * sr))
            calc_end = int(round(end * sr))
            print(f"  {word}: start={start:.3f}s={start_samp} samples (calc={calc_start}), "
                  f"end={end:.3f}s={end_samp} samples (calc={calc_end})")


if __name__ == "__main__":
    # Initialize generator
    generator = WordTimestampGenerator()
    
    print("=" * 80)
    print("WORD TIMESTAMPS COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Run all test categories
    test_basic_sentences()
    test_text_normalization()
    test_edge_cases()
    test_speaking_rates()
    test_rescaling_behavior()
    test_timeline_events()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)