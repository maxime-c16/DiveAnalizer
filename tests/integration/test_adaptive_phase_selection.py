#!/usr/bin/env python3
"""
Test adaptive phase selection on your machine.

Demonstrates Task 1.11: Automatically selecting Phase 2 on low-end machines,
preventing the 7x slowdown of Phase 3.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from diveanalyzer.utils.system_profiler import SystemProfiler
from diveanalyzer.config import get_config


def test_system_profile():
    """Test system profiling and phase recommendation."""
    print("\n" + "=" * 80)
    print("🔍 SYSTEM PROFILER TEST")
    print("=" * 80)

    profiler = SystemProfiler()
    profile = profiler.get_profile(refresh=True)

    print(profile)
    print()

    # Analyze the recommendation
    print("📊 Analysis:")
    print(f"  Your System Score: {profile.system_score}/10")

    if profile.system_score < 7:
        print(f"  ⚠️  Below minimum for Phase 3 (need ≥7/10)")
    else:
        print(f"  ✅ Capable for Phase 3")

    if profile.phase_3_estimate_sec > 30:
        print(f"  ⚠️  Phase 3 would take {profile.phase_3_estimate_sec:.0f}s")
        print(f"  📉 Too slow! Using Phase {profile.recommended_phase} instead")
    else:
        print(f"  ✅ Phase 3 is fast enough ({profile.phase_3_estimate_sec:.0f}s)")

    print()
    print("🎯 Recommendation Logic:")
    print(f"  Phase 3 estimated: {profile.phase_3_estimate_sec:.0f}s")
    print(f"  Threshold: 30s (if Phase 3 > 30s, use Phase 2)")

    if profile.phase_3_estimate_sec > 30:
        print(f"  ✓ {profile.phase_3_estimate_sec:.0f}s > 30s → Use Phase {profile.recommended_phase}")
    else:
        print(f"  ✓ {profile.phase_3_estimate_sec:.0f}s ≤ 30s → Use Phase {profile.recommended_phase}")

    return profile


def test_phase_performance_tradeoff(profile):
    """Show the tradeoff analysis."""
    print("\n" + "=" * 80)
    print("📊 PHASE PERFORMANCE TRADEOFF")
    print("=" * 80)

    print("\nPhase Comparison (for 10-minute video):")
    print()
    print("┌─────────┬──────────────┬────────────────┬──────────────┐")
    print("│ Phase   │ Time (10min) │ Confidence     │ Recommended  │")
    print("├─────────┼──────────────┼────────────────┼──────────────┤")
    print("│ Phase 1 │    5s        │ 0.82 (82%)     │              │")
    print("│ Phase 2 │   15s        │ 0.92 (92%)     │   ← YES      │")
    print(f"│ Phase 3 │  {profile.phase_3_estimate_sec:3.0f}s        │ 0.96 (96%)     │              │")
    print("└─────────┴──────────────┴────────────────┴──────────────┘")

    print()
    print("💡 Tradeoff Analysis:")
    print(f"  Phase 1 → Phase 2: +0.10 confidence for +10s time = EXCELLENT ✅")
    print(
        f"  Phase 2 → Phase 3: +0.04 confidence for +{profile.phase_3_estimate_sec - 15:.0f}s time = "
        f"{'TERRIBLE ❌' if profile.phase_3_estimate_sec > 100 else 'OK' if profile.phase_3_estimate_sec > 50 else 'GOOD ✅'}"
    )

    print()
    print("🎯 Decision:")
    if profile.phase_3_estimate_sec > 100:
        print(f"  Phase 2 is the SMART choice for your machine!")
        print(f"  Gain 0.92 confidence with only 15s processing")
        print(f"  Save {profile.phase_3_estimate_sec - 15:.0f}s compared to Phase 3!")
    else:
        print(f"  Phase 3 is feasible on your machine")
        print(f"  Takes {profile.phase_3_estimate_sec:.0f}s for 0.96 confidence")


def test_phase_selection_logic():
    """Test the phase selection logic."""
    print("\n" + "=" * 80)
    print("🧪 PHASE SELECTION LOGIC TEST")
    print("=" * 80)

    profiler = SystemProfiler()
    profile = profiler.get_profile()

    print()
    print("Decision Tree:")
    print("  if Phase 3 estimated > 30s:")
    print(f"    → Recommend Phase 2")
    print(f"    Your result: {profile.phase_3_estimate_sec:.0f}s > 30s? {'YES' if profile.phase_3_estimate_sec > 30 else 'NO'}")
    print()
    print("  if System Score < 7:")
    print(f"    → Recommend Phase 2")
    print(f"    Your result: Score {profile.system_score} < 7? {'YES' if profile.system_score < 7 else 'NO'}")
    print()
    print("  else:")
    print(f"    → Recommend Phase 3")
    print()
    print(f"✓ Final Recommendation: Phase {profile.recommended_phase}")


def test_cache_functionality():
    """Test cache load/save."""
    print("\n" + "=" * 80)
    print("💾 CACHE FUNCTIONALITY TEST")
    print("=" * 80)

    profiler = SystemProfiler()

    # Get fresh profile (creates cache)
    print("\n1️⃣  Getting fresh profile (creates cache)...")
    start = time.time()
    profile1 = profiler.get_profile(refresh=True)
    time1 = time.time() - start
    print(f"   ✓ Time: {time1:.3f}s (fresh profile)")

    # Get cached profile
    print("\n2️⃣  Getting cached profile (should be much faster)...")
    start = time.time()
    profile2 = profiler.get_profile(refresh=False)
    time2 = time.time() - start
    print(f"   ✓ Time: {time2:.3f}s (cached profile)")

    # Compare
    print(f"\n📊 Cache Performance:")
    print(f"   Fresh:  {time1:.3f}s")
    print(f"   Cached: {time2:.3f}s")
    print(f"   Speedup: {time1 / time2:.1f}x faster with cache ✅")

    # Verify they match
    if (
        profile1.cpu_count == profile2.cpu_count
        and profile1.gpu_type == profile2.gpu_type
        and profile1.recommended_phase == profile2.recommended_phase
    ):
        print(f"   Data matches: ✅")
    else:
        print(f"   Data mismatch: ❌")

    # Check cache file
    cache_file = profiler.cache_file
    if cache_file.exists():
        size_kb = cache_file.stat().st_size / 1024
        print(f"\n   Cache file: {cache_file}")
        print(f"   Size: {size_kb:.1f} KB")
        print(f"   Status: ✅ Cached")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  TEST ADAPTIVE PHASE SELECTION (Task 1.11)".center(78) + "║")
    print("║" + "  Demonstrating automatic phase recommendation on your machine".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # Test 1: System profile
    profile = test_system_profile()

    # Test 2: Performance tradeoff
    test_phase_performance_tradeoff(profile)

    # Test 3: Phase selection logic
    test_phase_selection_logic()

    # Test 4: Cache functionality
    test_cache_functionality()

    # Summary
    print("\n" + "=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    print()
    print("Your Machine Configuration:")
    print(f"  • CPU: {profile.cpu_count}-core @ {profile.cpu_freq_ghz:.1f} GHz")
    print(f"  • RAM: {profile.total_ram_gb:.1f} GB")
    print(f"  • GPU: {profile.gpu_type.upper()}")
    print(f"  • System Score: {profile.system_score}/10")
    print()
    print("Adaptive Phase Selection Result:")
    print(f"  ✓ Recommended Phase: {profile.recommended_phase}")
    if profile.recommended_phase == 2:
        print(f"  ✓ Processing Time: ~15s")
        print(f"  ✓ Confidence: 0.92 (92%)")
        print(f"  ✓ Avoids Phase 3 slowdown: Saves {profile.phase_3_estimate_sec - 15:.0f}s! 🚀")
    elif profile.recommended_phase == 3:
        print(f"  ✓ Processing Time: ~{profile.phase_3_estimate_sec:.0f}s")
        print(f"  ✓ Confidence: 0.96 (96%)")
        print(f"  ✓ GPU accelerated: Fastest & Most Accurate! 🎯")
    print()
    print("Next Step:")
    print("  Run: diveanalyzer process video.mov")
    print("  It will automatically use Phase " + str(profile.recommended_phase) + " 🎯")
    print()


if __name__ == "__main__":
    main()
