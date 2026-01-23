#!/usr/bin/env python3
"""Run tests against created test fixtures and generate required ground truth."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from diveanalyzer.detection.audio import extract_audio, detect_splash_peaks
from diveanalyzer.detection.motion import detect_motion_bursts
from diveanalyzer.detection.fusion import fuse_signals_audio_only, merge_overlapping_dives, filter_dives_by_confidence
from diveanalyzer.extraction.ffmpeg import get_video_duration, format_time


def test_short_dive():
    """Test Phase 1: Audio should find splash."""
    print("\n" + "="*70)
    print("TEST 1: short_dive.mp4 - Audio Detection")
    print("="*70)

    video = "tests/fixtures/short_dive.mp4"

    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        return None

    try:
        duration = get_video_duration(video)
        print(f"Duration: {format_time(duration)}")

        # Phase 1: Audio
        print("Running Phase 1 (audio detection)...")
        audio_path = extract_audio(video)
        peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)

        print(f"✓ Found {len(peaks)} splash peaks")
        for i, (time_s, amp) in enumerate(peaks, 1):
            print(f"  {i}. {time_s:6.2f}s (amplitude {amp:6.1f}dB)")

        # Phase 1 fusion
        dives = fuse_signals_audio_only(peaks)
        dives = merge_overlapping_dives(dives)
        dives = filter_dives_by_confidence(dives, min_confidence=0.5)

        print(f"✓ Fused to {len(dives)} dive(s)")
        for i, dive in enumerate(dives, 1):
            print(f"  Dive {i}: {dive.start_time:.1f}s - {dive.end_time:.1f}s (conf={dive.confidence:.2f})")

        return {
            "file": video,
            "status": "✓ PASS" if len(dives) >= 1 else "❌ FAIL",
            "peaks": len(peaks),
            "dives": len(dives),
            "expected_dives": 1,
            "confidence": dives[0].confidence if dives else 0.0,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"file": video, "status": f"❌ ERROR: {str(e)[:100]}", "peaks": None, "dives": None}


def test_multi_dive():
    """Test multiple dives."""
    print("\n" + "="*70)
    print("TEST 2: multi_dive.mp4 - Multiple Dives")
    print("="*70)

    video = "tests/fixtures/multi_dive.mp4"

    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        return None

    try:
        duration = get_video_duration(video)
        print(f"Duration: {format_time(duration)}")

        # Phase 1: Audio
        print("Running Phase 1 (audio detection)...")
        audio_path = extract_audio(video)
        peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)

        print(f"✓ Found {len(peaks)} splash peaks")
        for i, (time_s, amp) in enumerate(peaks[:5], 1):  # Show first 5
            print(f"  {i}. {time_s:6.2f}s")
        if len(peaks) > 5:
            print(f"  ... and {len(peaks)-5} more")

        # Fuse
        dives = fuse_signals_audio_only(peaks)
        dives = merge_overlapping_dives(dives)
        dives = filter_dives_by_confidence(dives, min_confidence=0.5)

        print(f"✓ Fused to {len(dives)} dive(s)")
        for i, dive in enumerate(dives[:5], 1):
            print(f"  Dive {i}: {dive.start_time:.1f}s - {dive.end_time:.1f}s (conf={dive.confidence:.2f})")
        if len(dives) > 5:
            print(f"  ... and {len(dives)-5} more")

        return {
            "file": video,
            "status": "✓ PASS" if len(dives) >= 3 else f"⚠️  WARN (expected ≥3, got {len(dives)})",
            "peaks": len(peaks),
            "dives": len(dives),
            "expected_dives": "3+",
            "confidence": sum(d.confidence for d in dives) / len(dives) if dives else 0.0,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"file": video, "status": f"❌ ERROR: {str(e)[:100]}", "peaks": None, "dives": None}


def test_no_audio():
    """Test Phase 2/3: Should work without audio."""
    print("\n" + "="*70)
    print("TEST 3: no_audio.mp4 - Motion/Person Detection")
    print("="*70)

    video = "tests/fixtures/edge_cases/no_audio.mp4"

    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        return None

    try:
        duration = get_video_duration(video)
        print(f"Duration: {format_time(duration)}")

        # Try to extract audio (should fail or be silent)
        print("Attempting audio extraction (expected to be silent)...")
        try:
            audio_path = extract_audio(video)
            peaks = detect_splash_peaks(audio_path, threshold_db=-25.0)
            print(f"⚠️  Found {len(peaks)} audio peaks (unexpected, should be silent)")
        except Exception as audio_err:
            peaks = []
            print(f"⚠️  Audio extraction issue: {str(audio_err)[:50]} (expected)")

        # Try motion
        print("Testing Phase 2 (motion detection)...")
        try:
            motion_events = detect_motion_bursts(video, sample_fps=5.0)
            print(f"✓ Found {len(motion_events)} motion bursts")
            return {
                "file": video,
                "status": "✓ PASS" if len(motion_events) >= 1 else "❌ FAIL",
                "audio_peaks": len(peaks),
                "motion_events": len(motion_events),
                "expected": "motion detection should work",
                "note": "Audio is missing, relied on motion",
            }
        except Exception as motion_err:
            print(f"⚠️  Motion detection error: {str(motion_err)[:100]}")
            return {
                "file": video,
                "status": f"⚠️  WARN - motion failed",
                "audio_peaks": len(peaks),
                "motion_events": None,
                "error": str(motion_err)[:100],
            }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"file": video, "status": f"❌ ERROR: {str(e)[:100]}"}


def test_false_positive():
    """Test that loud noise without splash is rejected."""
    print("\n" + "="*70)
    print("TEST 4: false_positive.mp4 - Should NOT detect as dive")
    print("="*70)

    video = "tests/fixtures/edge_cases/false_positive.mp4"

    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        return None

    try:
        duration = get_video_duration(video)
        print(f"Duration: {format_time(duration)}")

        print("Running Phase 1 (audio detection)...")
        audio_path = extract_audio(video)
        peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)

        print(f"⚠️  Found {len(peaks)} audio peaks")
        for i, (time_s, amp) in enumerate(peaks, 1):
            print(f"  {i}. {time_s:6.2f}s (amplitude {amp:6.1f}dB)")

        # Check if Phase 2 rejects it
        if len(peaks) > 0:
            dives = fuse_signals_audio_only(peaks)
            dives = merge_overlapping_dives(dives)
            dives = filter_dives_by_confidence(dives, min_confidence=0.5)
            print(f"⚠️  Phase 1 created {len(dives)} events (motion/person should reject)")
        else:
            dives = []

        return {
            "file": video,
            "status": "✓ PASS (no dives)" if len(dives) == 0 else "⚠️  WARN (audio peaks found, needs Phase 2/3 validation)",
            "audio_peaks": len(peaks),
            "dives_from_audio": len(dives),
            "expected": 0,
            "note": "Should be rejected by motion/person validation if present",
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"file": video, "status": f"❌ ERROR: {str(e)[:100]}"}


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  TEST FIXTURE VALIDATION".center(68) + "║")
    print("║" + "  Running DiveAnalyzer against created test videos".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    results = []

    # Run tests
    results.append(test_short_dive())
    results.append(test_multi_dive())
    results.append(test_no_audio())
    results.append(test_false_positive())

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print()

    for r in results:
        if r is None:
            continue
        print(f"{r['status']:20s} | {Path(r['file']).name:25s} | ", end="")
        if "peaks" in r and r["peaks"] is not None:
            print(f"Peaks: {r.get('peaks', '?')}, ", end="")
        if "dives" in r and r["dives"] is not None:
            print(f"Dives: {r.get('dives', '?')}, Expected: {r.get('expected_dives', '?')}")
        else:
            print()

    print()
    print("="*70)
    print("GROUND TRUTH DATA NEEDED")
    print("="*70)
    print("""
For complete test coverage, we need to create a ground_truth.json file with
manual annotations for each test video. This file should contain:

{
  "short_dive.mp4": {
    "duration_sec": 40,
    "dives": [
      {
        "id": 1,
        "start_time": XX.X,
        "end_time": XX.X,
        "splash_time": XX.X,
        "type": "dive"  # or "motion_burst" or "person_absent"
      }
    ]
  },
  "multi_dive.mp4": {
    "duration_sec": 300,
    "dives": [
      { "id": 1, "start_time": XX.X, "end_time": XX.X, "splash_time": XX.X },
      { "id": 2, "start_time": XX.X, "end_time": XX.X, "splash_time": XX.X },
      { "id": 3, "start_time": XX.X, "end_time": XX.X, "splash_time": XX.X }
    ]
  },
  "edge_cases/no_audio.mp4": {
    "duration_sec": 40,
    "dives": [...],
    "note": "No audio track - should detect via motion/person"
  },
  "edge_cases/false_positive.mp4": {
    "duration_sec": 30,
    "dives": [],
    "note": "Loud noise but no actual dive"
  },
  ...
}

TODO: You need to manually watch each video and record:
  1. Exact start/end times for each dive
  2. Exact splash time (when person hits water)
  3. Any special notes (missing audio, false positives, etc)
""")


if __name__ == "__main__":
    main()
