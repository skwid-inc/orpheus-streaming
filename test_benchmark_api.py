#!/usr/bin/env python3
"""
Test script for the benchmark API endpoint.
"""

import requests
import json
import sys

def test_benchmark_api(base_url="http://localhost:9090"):
    """Test the benchmark API endpoint."""
    
    print("Testing Orpheus TTS Benchmark API")
    print("=" * 50)
    
    # Test payload
    benchmark_request = {
        "text": "Hello, this is a test of the Orpheus text to speech system. We are testing the performance metrics including time to first byte and real-time factor.",
        "voice": "tara",
        "num_runs": 3,
        "warmup": True
    }
    
    print(f"\nRequest payload:")
    print(json.dumps(benchmark_request, indent=2))
    
    try:
        # Send request
        print(f"\nSending benchmark request to {base_url}/v1/benchmark...")
        response = requests.post(
            f"{base_url}/v1/benchmark",
            json=benchmark_request,
            timeout=120  # 2 minutes timeout for benchmark
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nBenchmark completed successfully!")
            print(f"\nResults:")
            print(f"- Text length: {result['text_length']} characters")
            print(f"- Voice: {result['voice']}")
            print(f"- Successful runs: {result['successful_runs']}")
            print(f"- Failed runs: {result['failed_runs']}")
            
            metrics = result.get('metrics', {})
            
            # TTFB metrics
            if 'ttfb' in metrics:
                ttfb = metrics['ttfb']
                print(f"\nTime to First Byte (TTFB):")
                print(f"  - Mean: {ttfb['mean_ms']:.2f} ms")
                print(f"  - StdDev: {ttfb['stddev_ms']:.2f} ms")
                print(f"  - Min: {ttfb['min_ms']:.2f} ms")
                print(f"  - Max: {ttfb['max_ms']:.2f} ms")
            
            # Total time metrics
            if 'total_time' in metrics:
                total = metrics['total_time']
                print(f"\nTotal Generation Time:")
                print(f"  - Mean: {total['mean_seconds']:.3f} s")
                print(f"  - StdDev: {total['stddev_seconds']:.3f} s")
                print(f"  - Min: {total['min_seconds']:.3f} s")
                print(f"  - Max: {total['max_seconds']:.3f} s")
            
            # RTF metrics
            if 'rtf' in metrics:
                rtf = metrics['rtf']
                print(f"\nReal-Time Factor (RTF):")
                print(f"  - Mean: {rtf['mean']:.3f}")
                print(f"  - StdDev: {rtf['stddev']:.3f}")
                print(f"  - Min: {rtf['min']:.3f}")
                print(f"  - Max: {rtf['max']:.3f}")
                print(f"  - {rtf['description']}")
            
            # Audio info
            if 'audio' in metrics:
                audio = metrics['audio']
                print(f"\nAudio Characteristics:")
                print(f"  - Sample Rate: {audio['sample_rate']} Hz")
                print(f"  - Bits per Sample: {audio['bits_per_sample']}")
                print(f"  - Channels: {audio['channels']}")
                print(f"  - Average Duration: {audio['average_duration_seconds']:.3f} s")
                
        else:
            print(f"\nError: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("\nError: Request timed out")
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to server. Is it running?")
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    # Allow custom URL if provided
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9090"
    test_benchmark_api(url)
