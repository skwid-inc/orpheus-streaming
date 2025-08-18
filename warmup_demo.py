#!/usr/bin/env python3
"""
Demo script to show the impact of SNAC warmup on performance.

This script compares performance with and without warmup to demonstrate
the cold start penalty elimination.

Usage:
    python warmup_demo.py
"""

import os
import time
import logging
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_without_warmup():
    """Test TTS performance without warmup (cold start)."""
    print("\n🥶 Testing WITHOUT warmup (cold start)...")
    
    # Disable automatic warmup
    os.environ["ENABLE_SNAC_WARMUP"] = "false"
    
    # Import after setting env var to prevent auto-warmup
    from src.trt_engine import OrpheusModelTRT
    
    model = OrpheusModelTRT()
    
    # Test inference (this will hit the cold start penalty)
    test_prompt = "Hello world"
    test_voice = "tara"
    
    start_time = time.perf_counter()
    audio_chunks = []
    
    async for audio_chunk in model.generate_speech_async(test_prompt, test_voice):
        audio_chunks.append(audio_chunk)
        if len(audio_chunks) >= 3:  # Just get a few chunks
            break
    
    total_time = time.perf_counter() - start_time
    total_audio = sum(len(chunk) for chunk in audio_chunks)
    
    print(f"❄️  Cold start performance:")
    print(f"   Time: {total_time*1000:.2f}ms")
    print(f"   Audio: {len(audio_chunks)} chunks, {total_audio} bytes")
    
    return total_time

async def test_with_warmup():
    """Test TTS performance with warmup."""
    print("\n🔥 Testing WITH warmup...")
    
    # Enable automatic warmup
    os.environ["ENABLE_SNAC_WARMUP"] = "true"
    
    # Force reload of decoder module to trigger warmup
    import importlib
    import src.decoder
    importlib.reload(src.decoder)
    
    from src.trt_engine import OrpheusModelTRT
    
    model = OrpheusModelTRT()
    
    # Additional explicit warmup (models are already warmed from import)
    model.warmup_models()
    
    # Test inference (should be fast now)
    test_prompt = "Hello world"
    test_voice = "tara"
    
    start_time = time.perf_counter()
    audio_chunks = []
    
    async for audio_chunk in model.generate_speech_async(test_prompt, test_voice):
        audio_chunks.append(audio_chunk)
        if len(audio_chunks) >= 3:  # Just get a few chunks
            break
    
    total_time = time.perf_counter() - start_time
    total_audio = sum(len(chunk) for chunk in audio_chunks)
    
    print(f"🚀 Warmed up performance:")
    print(f"   Time: {total_time*1000:.2f}ms")
    print(f"   Audio: {len(audio_chunks)} chunks, {total_audio} bytes")
    
    return total_time

async def main():
    """Run the warmup demonstration."""
    print("🎯 SNAC Warmup Performance Demo")
    print("=" * 50)
    
    try:
        # Test cold start performance
        cold_time = await test_without_warmup()
        
        # Test warmed up performance
        warm_time = await test_with_warmup()
        
        # Calculate improvement
        improvement = (cold_time - warm_time) / cold_time * 100
        speedup = cold_time / warm_time
        
        print(f"\n📊 Performance Comparison:")
        print(f"   Cold start: {cold_time*1000:.2f}ms")
        print(f"   Warmed up:  {warm_time*1000:.2f}ms")
        print(f"   Improvement: {improvement:.1f}% faster")
        print(f"   Speedup: {speedup:.1f}x")
        
        if improvement > 20:
            print("✅ Significant performance improvement achieved!")
        else:
            print("ℹ️  Modest improvement - warmup may already be effective")
            
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure your environment is set up correctly with all dependencies.")

if __name__ == "__main__":
    asyncio.run(main())
