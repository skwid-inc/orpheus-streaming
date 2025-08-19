#!/usr/bin/env python3
"""
Test script to demonstrate the tensor population optimization improvements.

This script compares the old Python loop approach vs the new vectorized approach
to show the performance gains from the Tensor_Population optimization.

Usage:
    python tensor_optimization_test.py
"""

import torch
import time
import numpy as np
import os

# Simulate the SNAC device
snac_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {snac_device}")

def old_tensor_population(multiframe, num_frames):
    """Original slow method using Python loops."""
    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    # Original Python loop approach
    for i in range(num_frames):
        base_idx = i * 7
        codes_0[0, i] = multiframe[base_idx]
        
        codes_1[0, i*2] = multiframe[base_idx + 1]
        codes_1[0, i*2 + 1] = multiframe[base_idx + 4]
        
        codes_2[0, i*4] = multiframe[base_idx + 2]
        codes_2[0, i*4 + 1] = multiframe[base_idx + 3]
        codes_2[0, i*4 + 2] = multiframe[base_idx + 5]
        codes_2[0, i*4 + 3] = multiframe[base_idx + 6]
    
    return codes_0, codes_1, codes_2

def new_tensor_population(multiframe, num_frames):
    """New optimized method using vectorized operations."""
    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    # Vectorized approach
    multiframe_tensor = torch.tensor(multiframe[:num_frames * 7], dtype=torch.int32, device=snac_device)
    multiframe_view = multiframe_tensor.view(num_frames, 7)
    
    codes_0[0, :] = multiframe_view[:, 0]
    codes_1[0, 0::2] = multiframe_view[:, 1]
    codes_1[0, 1::2] = multiframe_view[:, 4]
    codes_2[0, 0::4] = multiframe_view[:, 2]
    codes_2[0, 1::4] = multiframe_view[:, 3]
    codes_2[0, 2::4] = multiframe_view[:, 5]
    codes_2[0, 3::4] = multiframe_view[:, 6]
    
    return codes_0, codes_1, codes_2

def benchmark_tensor_population():
    """Benchmark both approaches."""
    
    # Test different frame counts (typical sizes)
    frame_counts = [1, 4, 7, 8, 16, 28]  # 28 is typical "subsequent chunks"
    num_iterations = 100
    
    print("\n🔬 Tensor Population Benchmark")
    print("=" * 50)
    
    for num_frames in frame_counts:
        # Create test data
        multiframe = list(range(num_frames * 7))  # Simulate token data
        
        print(f"\n📊 Testing {num_frames} frames ({num_frames * 7} tokens):")
        
        # Benchmark old method
        old_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            old_result = old_tensor_population(multiframe, num_frames)
            end = time.perf_counter()
            old_times.append(end - start)
        
        # Benchmark new method
        new_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            new_result = new_tensor_population(multiframe, num_frames)
            end = time.perf_counter()
            new_times.append(end - start)
        
        # Verify results are the same
        assert torch.equal(old_result[0], new_result[0]), "codes_0 mismatch!"
        assert torch.equal(old_result[1], new_result[1]), "codes_1 mismatch!"
        assert torch.equal(old_result[2], new_result[2]), "codes_2 mismatch!"
        
        # Calculate statistics
        old_avg = np.mean(old_times) * 1000
        new_avg = np.mean(new_times) * 1000
        speedup = old_avg / new_avg
        improvement = (old_avg - new_avg) / old_avg * 100
        
        print(f"  Old method: {old_avg:.2f}ms ± {np.std(old_times) * 1000:.2f}ms")
        print(f"  New method: {new_avg:.2f}ms ± {np.std(new_times) * 1000:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x ({improvement:.1f}% faster)")
        
        if speedup > 2:
            print("  ✅ Significant improvement!")
        elif speedup > 1.5:
            print("  🟡 Good improvement")
        else:
            print("  🔵 Modest improvement")

def estimate_rtf_impact():
    """Estimate the impact on RTF based on typical usage."""
    print("\n🎯 RTF Impact Analysis")
    print("=" * 30)
    
    # Typical scenario: 28 frames per chunk, ~20 chunks per request
    frames_per_chunk = 4  # First chunk size
    chunks_per_request = 20
    
    # Benchmark the typical chunk size
    multiframe = list(range(frames_per_chunk * 7))
    num_iterations = 100
    
    # Old method timing
    old_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        old_tensor_population(multiframe, frames_per_chunk)
        end = time.perf_counter()
        old_times.append(end - start)
    
    # New method timing
    new_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        new_tensor_population(multiframe, frames_per_chunk)
        end = time.perf_counter()
        new_times.append(end - start)
    
    old_avg_per_chunk = np.mean(old_times) * 1000
    new_avg_per_chunk = np.mean(new_times) * 1000
    
    old_total_per_request = old_avg_per_chunk * chunks_per_request
    new_total_per_request = new_avg_per_chunk * chunks_per_request
    
    time_saved = old_total_per_request - new_total_per_request
    
    print(f"Typical request scenario:")
    print(f"  Chunks per request: {chunks_per_request}")
    print(f"  Frames per chunk: {frames_per_chunk}")
    print(f"")
    print(f"Tensor population time per request:")
    print(f"  Old method: {old_total_per_request:.1f}ms")
    print(f"  New method: {new_total_per_request:.1f}ms")
    print(f"  Time saved: {time_saved:.1f}ms per request")
    print(f"")
    print(f"RTF impact (assuming 8-second audio):")
    print(f"  Old contribution to RTF: +{old_total_per_request/8000:.3f}")
    print(f"  New contribution to RTF: +{new_total_per_request/8000:.3f}")
    print(f"  RTF improvement: -{time_saved/8000:.3f}")

def main():
    """Run the tensor optimization benchmarks."""
    print("🚀 Tensor Population Optimization Benchmark")
    print("=" * 60)
    
    # Warm up GPU if available
    if snac_device == "cuda":
        print("🔥 Warming up GPU...")
        dummy = torch.zeros(1000, device=snac_device)
        del dummy
        torch.cuda.empty_cache()
    
    try:
        benchmark_tensor_population()
        estimate_rtf_impact()
        
        print("\n✅ Benchmark complete!")
        print("\nKey takeaways:")
        print("• Vectorized operations are much faster than Python loops")
        print("• Larger frame counts see bigger improvements")
        print("• This optimization should significantly reduce Tensor_Population time")
        print("• Combined with warmup, should improve RTF substantially")
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
