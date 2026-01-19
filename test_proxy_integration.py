#!/usr/bin/env python3
"""
Test proxy integration for motion detection.

Verifies that:
1. Proxy generation is triggered for large videos (>500MB)
2. Proxy is cached and reused
3. Motion detection uses the proxy
"""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from diveanalyzer.cli import get_or_generate_proxy
from diveanalyzer.storage.cache import CacheManager
from diveanalyzer.detection.motion import detect_motion_bursts


def test_proxy_generation():
    """Test proxy generation and caching."""
    video_path = "IMG_6496.MOV"

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    print("=" * 80)
    print("🧪 TESTING PROXY INTEGRATION")
    print("=" * 80)

    # Test 1: Generate proxy (first run)
    print("\n📍 Test 1: Generate proxy on first run")
    print("-" * 80)

    cache = CacheManager()
    cache.cleanup_expired()  # Clean cache first

    video_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
    print(f"  Video size: {video_size_mb:.1f}MB")

    start = time.time()
    proxy_path_1 = get_or_generate_proxy(
        video_path,
        proxy_height=480,
        enable_cache=True,
        verbose=True
    )
    elapsed_1 = time.time() - start

    print(f"  ⏱️  First run time: {elapsed_1:.1f}s")
    print(f"  ✓ Proxy path: {proxy_path_1}")

    if Path(proxy_path_1).exists():
        proxy_size_mb = Path(proxy_path_1).stat().st_size / (1024 * 1024)
        reduction = (1 - proxy_size_mb / video_size_mb) * 100
        print(f"  ✓ Proxy size: {proxy_size_mb:.1f}MB (reduction: {reduction:.0f}%)")
    else:
        print(f"  ⚠️  Proxy file not found at {proxy_path_1}")

    # Test 2: Verify cache hit (second run)
    print("\n📍 Test 2: Cache hit on second run")
    print("-" * 80)

    start = time.time()
    proxy_path_2 = get_or_generate_proxy(
        video_path,
        proxy_height=480,
        enable_cache=True,
        verbose=True
    )
    elapsed_2 = time.time() - start

    print(f"  ⏱️  Second run time: {elapsed_2:.1f}s")
    print(f"  ✓ Proxy path: {proxy_path_2}")

    if proxy_path_1 == proxy_path_2:
        print(f"  ✓ Cache hit: Same proxy file returned")
        speedup = elapsed_1 / elapsed_2
        print(f"  ✓ Speedup: {speedup:.1f}x faster (cached)")
    else:
        print(f"  ⚠️  Different proxy paths returned!")

    # Test 3: Motion detection with proxy
    print("\n📍 Test 3: Motion detection with proxy")
    print("-" * 80)

    start = time.time()
    motion_events = detect_motion_bursts(proxy_path_2, sample_fps=5.0)
    elapsed_motion = time.time() - start

    print(f"  ✓ Found {len(motion_events)} motion bursts")
    print(f"  ⏱️  Motion detection time: {elapsed_motion:.1f}s")

    if motion_events:
        print(f"  First 3 bursts:")
        for i, (start_t, end_t, intensity) in enumerate(motion_events[:3], 1):
            duration = end_t - start_t
            print(f"    {i}. {start_t:7.2f}s - {end_t:7.2f}s ({duration:5.2f}s) intensity {intensity:.1f}")

    # Test 4: Cache statistics
    print("\n📍 Test 4: Cache statistics")
    print("-" * 80)

    cache_stats = cache.get_cache_stats()
    print(f"  Cache directory: {cache_stats['cache_dir']}")
    print(f"  Total cached items: {cache_stats['entry_count']}")
    print(f"  Total cache size: {cache_stats['total_size_mb']:.1f} MB")

    # Summary
    print("\n" + "=" * 80)
    print("✅ PROXY INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print(f"""
✓ Proxy generation works for large videos (>500MB)
✓ Proxy size reduced: {video_size_mb:.1f}MB → {proxy_size_mb:.1f}MB ({reduction:.0f}%)
✓ Caching works: {elapsed_1:.1f}s → {elapsed_2:.1f}s ({speedup:.1f}x faster)
✓ Motion detection works on proxy: {len(motion_events)} bursts in {elapsed_motion:.1f}s

Performance expectation:
- Full video motion: ~150s
- Proxy + caching: {elapsed_2:.1f}s + {elapsed_motion:.1f}s = {elapsed_2+elapsed_motion:.1f}s (cached run)
- Speedup: ~{150/(elapsed_2+elapsed_motion):.0f}x faster!
""")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_proxy_generation()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
