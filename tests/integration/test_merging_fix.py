#!/usr/bin/env python3
"""Test the merging bug fix on multi_dive.mp4."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from diveanalyzer.detection.audio import extract_audio, detect_splash_peaks
from diveanalyzer.detection.fusion import fuse_signals_audio_only, merge_overlapping_dives
from diveanalyzer.extraction.ffmpeg import get_video_duration, format_time


def test_multi_dive_merging():
    """Test the merging fix on multi_dive.mp4."""
    video = "tests/fixtures/multi_dive.mp4"

    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        return

    print("\n" + "=" * 80)
    print("TESTING MERGING BUG FIX: multi_dive.mp4")
    print("=" * 80)

    duration = get_video_duration(video)
    print(f"\nVideo duration: {format_time(duration)}")

    # Phase 1: Extract audio and detect peaks
    print("\n🔊 Phase 1: Audio Detection")
    print("─" * 80)

    audio_path = extract_audio(video)
    peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)

    print(f"✓ Found {len(peaks)} audio peaks\n")

    # Show first and last peaks
    if len(peaks) > 10:
        print("First 5 peaks:")
        for i, (time_s, amp) in enumerate(peaks[:5], 1):
            print(f"  {i}. {time_s:6.2f}s (amplitude {amp:6.1f}dB)")
        print(f"  ... ({len(peaks) - 10} more) ...")
        print("Last 5 peaks:")
        for i, (time_s, amp) in enumerate(peaks[-5:], len(peaks) - 4):
            print(f"  {i}. {time_s:6.2f}s (amplitude {amp:6.1f}dB)")
    else:
        for i, (time_s, amp) in enumerate(peaks, 1):
            print(f"  {i}. {time_s:6.2f}s (amplitude {amp:6.1f}dB)")

    # Fuse into initial dives
    print("\n📊 After Fusion (BEFORE merging):")
    print("─" * 80)
    dives = fuse_signals_audio_only(peaks)
    print(f"✓ Created {len(dives)} initial dive events from {len(peaks)} peaks")

    # Test the OLD behavior (using start_time for merge)
    # This is what would happen with the bug
    def old_merge_logic(dives_list):
        """Simulate old buggy merge logic for comparison."""
        if not dives_list:
            return []
        dives_list = sorted(dives_list, key=lambda d: d.start_time)
        merged = [dives_list[0]]
        for dive in dives_list[1:]:
            prev = merged[-1]
            if dive.start_time < prev.end_time + 5.0:  # OLD: using start_time
                prev.end_time = max(prev.end_time, dive.end_time)
                prev.confidence = max(prev.confidence, dive.confidence)
            else:
                merged.append(dive)
        return merged

    print("\n📊 With OLD (buggy) merging logic:")
    print("─" * 80)
    old_merged = old_merge_logic(dives.copy())
    print(f"❌ OLD behavior: {len(peaks)} peaks → {len(old_merged)} merged dive(s)")
    if len(old_merged) > 0:
        print(f"   Single mega-dive from {old_merged[0].start_time:.1f}s to {old_merged[0].end_time:.1f}s")

    # Apply the NEW fixed merging logic
    print("\n📊 With NEW (fixed) merging logic:")
    print("─" * 80)
    merged_dives = merge_overlapping_dives(dives)
    print(f"✓ NEW behavior: {len(peaks)} peaks → {len(merged_dives)} merged dive(s)")

    # Show details of merged dives
    if len(merged_dives) > 5:
        print("\nFirst 5 merged dives:")
        for dive in merged_dives[:5]:
            duration_sec = dive.end_time - dive.start_time
            print(f"  Dive {dive.dive_number}: splash@{dive.splash_time:6.2f}s ({duration_sec:5.1f}s duration, conf={dive.confidence:.2f})")
        print(f"  ... ({len(merged_dives) - 5} more) ...")
    else:
        print("\nMerged dives:")
        for dive in merged_dives:
            duration_sec = dive.end_time - dive.start_time
            print(f"  Dive {dive.dive_number}: splash@{dive.splash_time:6.2f}s ({duration_sec:5.1f}s duration, conf={dive.confidence:.2f})")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print(f"\n✓ Input: {len(peaks)} audio peaks detected")
    print(f"  → Old (buggy) logic: {len(old_merged)} merged event ❌")
    print(f"  → New (fixed) logic: {len(merged_dives)} merged events ✓")

    if len(merged_dives) > 1:
        print(f"\n✅ SUCCESS: Merging fix works! Multiple dives detected instead of single mega-event.")
        print(f"\nDive separation details:")
        for i, dive in enumerate(merged_dives, 1):
            gap_from_prev = (
                "—" if i == 1
                else f"{dive.splash_time - merged_dives[i-2].splash_time:.2f}s gap"
            )
            print(f"  Dive {i}: splash@{dive.splash_time:6.2f}s {gap_from_prev}")
    else:
        print(f"\n⚠️  Only {len(merged_dives)} dive detected. May need ground truth validation.")
        print("    Please provide ground truth timestamps from GROUND_TRUTH_TEMPLATE.json")

    # Show gaps between merges
    if len(merged_dives) > 1:
        print(f"\nGaps between consecutive dives:")
        for i in range(1, len(merged_dives)):
            gap = merged_dives[i].splash_time - merged_dives[i-1].splash_time
            print(f"  {merged_dives[i-1].dive_number} → {merged_dives[i].dive_number}: {gap:.2f}s")


if __name__ == "__main__":
    test_multi_dive_merging()
