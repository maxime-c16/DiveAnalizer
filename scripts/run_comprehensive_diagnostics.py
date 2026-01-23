#!/usr/bin/env python3
"""Comprehensive diagnostics on all test fixtures across all phases."""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent))

from diveanalyzer.detection.audio import extract_audio, detect_splash_peaks
from diveanalyzer.detection.motion import detect_motion_bursts
from diveanalyzer.detection.person import detect_person_frames
from diveanalyzer.detection.fusion import (
    fuse_signals_audio_only,
    fuse_signals_audio_motion,
    fuse_signals_audio_motion_person,
    merge_overlapping_dives,
    filter_dives_by_confidence,
)
from diveanalyzer.extraction.ffmpeg import get_video_duration, format_time


def run_phase_1_analysis(video_path: str) -> Dict[str, Any]:
    """Run Phase 1 (audio-only) analysis."""
    try:
        audio_path = extract_audio(video_path)
        peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)
        dives = fuse_signals_audio_only(peaks)
        dives = merge_overlapping_dives(dives)
        filtered = filter_dives_by_confidence(dives, min_confidence=0.5)

        return {
            "status": "✓",
            "peaks": len(peaks),
            "fused": len(dives),
            "filtered": len(filtered),
            "avg_confidence": sum(d.confidence for d in filtered) / len(filtered) if filtered else 0.0,
            "dives": [
                {
                    "id": i + 1,
                    "start": d.start_time,
                    "end": d.end_time,
                    "confidence": d.confidence,
                }
                for i, d in enumerate(filtered)
            ],
        }
    except Exception as e:
        return {"status": "❌", "error": str(e)[:100]}


def run_phase_2_analysis(video_path: str) -> Dict[str, Any]:
    """Run Phase 2 (audio + motion) analysis."""
    try:
        # Phase 1 first
        audio_path = extract_audio(video_path)
        peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)

        # Phase 2
        motion_events = detect_motion_bursts(video_path, sample_fps=5.0)
        dives = fuse_signals_audio_motion(peaks, motion_events)
        dives = merge_overlapping_dives(dives)
        filtered = filter_dives_by_confidence(dives, min_confidence=0.5)

        return {
            "status": "✓",
            "audio_peaks": len(peaks),
            "motion_bursts": len(motion_events),
            "fused": len(dives),
            "filtered": len(filtered),
            "avg_confidence": sum(d.confidence for d in filtered) / len(filtered) if filtered else 0.0,
            "dives": [
                {
                    "id": i + 1,
                    "start": d.start_time,
                    "end": d.end_time,
                    "confidence": d.confidence,
                }
                for i, d in enumerate(filtered)
            ],
        }
    except Exception as e:
        return {"status": "⚠️", "error": str(e)[:100]}


def run_phase_3_analysis(video_path: str) -> Dict[str, Any]:
    """Run Phase 3 (audio + motion + person) analysis."""
    try:
        # Phase 1
        audio_path = extract_audio(video_path)
        peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)

        # Phase 2
        motion_events = detect_motion_bursts(video_path, sample_fps=5.0)

        # Phase 3
        person_events = detect_person_frames(
            video_path,
            sample_fps=5.0,
            use_gpu=False,  # Force CPU for compatibility
            force_cpu=True,
        )

        dives = fuse_signals_audio_motion_person(peaks, motion_events, person_events)
        dives = merge_overlapping_dives(dives)
        filtered = filter_dives_by_confidence(dives, min_confidence=0.5)

        person_frame_count = sum(1 for _, person_detected, _ in person_events if person_detected)

        return {
            "status": "✓",
            "audio_peaks": len(peaks),
            "motion_bursts": len(motion_events),
            "person_frames": person_frame_count,
            "total_frames_checked": len(person_events),
            "fused": len(dives),
            "filtered": len(filtered),
            "avg_confidence": sum(d.confidence for d in filtered) / len(filtered) if filtered else 0.0,
            "dives": [
                {
                    "id": i + 1,
                    "start": d.start_time,
                    "end": d.end_time,
                    "confidence": d.confidence,
                }
                for i, d in enumerate(filtered)
            ],
        }
    except Exception as e:
        return {"status": "⚠️", "error": str(e)[:100]}


def main():
    """Run diagnostics on all test fixtures."""
    fixtures = [
        ("short_dive.mp4", 1, "Single dive, 40s"),
        ("multi_dive.mp4", 3, "Multiple dives, 5min (CRITICAL - merging bug)"),
        ("noisy_audio.mp4", 1, "Noisy background, 90s"),
        ("edge_cases/back_to_back.mp4", 2, "Two dives <5s apart"),
        ("edge_cases/no_audio.mp4", 1, "No audio track, fallback to Phase 2"),
        ("edge_cases/failed_dive.mp4", 0, "Motion but no splash (abort)"),
        ("edge_cases/false_positive.mp4", 0, "Loud noise but NO dive"),
        ("edge_cases/very_short.mp4", 1, "3-second clip"),
    ]

    fixture_dir = Path("tests/fixtures")
    results = {}

    print("\n" + "=" * 90)
    print("COMPREHENSIVE PHASE DIAGNOSTICS ON TEST FIXTURES")
    print("=" * 90)

    for fixture_name, expected_dives, description in fixtures:
        fixture_path = fixture_dir / fixture_name

        if not fixture_path.exists():
            print(f"\n❌ {fixture_name}: FILE NOT FOUND")
            continue

        print(f"\n{'─' * 90}")
        print(f"📹 {fixture_name}")
        print(f"   Expected: {expected_dives} dive(s) | {description}")
        print(f"{'─' * 90}")

        try:
            duration = get_video_duration(str(fixture_path))
            print(f"   Duration: {format_time(duration)}")
        except Exception as e:
            print(f"   ⚠️  Could not get duration: {e}")
            continue

        fixture_results = {}

        # Phase 1
        print(f"\n   🔊 PHASE 1 (Audio only):")
        phase1 = run_phase_1_analysis(str(fixture_path))
        fixture_results["phase_1"] = phase1

        if phase1["status"] == "✓":
            print(f"      Peaks: {phase1['peaks']} → Fused: {phase1['fused']} → Filtered: {phase1['filtered']}")
            print(f"      Avg confidence: {phase1['avg_confidence']:.2f}")
            if phase1["filtered"] > 0:
                print(f"      Dives: {phase1['dives']}")
            if phase1["filtered"] != expected_dives:
                print(f"      ⚠️  Expected {expected_dives}, got {phase1['filtered']}")
        else:
            print(f"      ❌ Error: {phase1['error']}")

        # Phase 2
        print(f"\n   🎬 PHASE 2 (Audio + Motion):")
        phase2 = run_phase_2_analysis(str(fixture_path))
        fixture_results["phase_2"] = phase2

        if phase2["status"] in ["✓", "⚠️"]:
            if "error" in phase2:
                print(f"      ⚠️  {phase2['error']}")
            else:
                print(f"      Audio peaks: {phase2['audio_peaks']} | Motion bursts: {phase2['motion_bursts']}")
                print(f"      Fused: {phase2['fused']} → Filtered: {phase2['filtered']}")
                print(f"      Avg confidence: {phase2['avg_confidence']:.2f}")
                if phase2["filtered"] > 0:
                    print(f"      Dives: {phase2['dives']}")
                if phase2["filtered"] != expected_dives:
                    print(f"      ⚠️  Expected {expected_dives}, got {phase2['filtered']}")
        else:
            print(f"      ❌ Error: {phase2['error']}")

        # Phase 3
        print(f"\n   👤 PHASE 3 (Audio + Motion + Person):")
        phase3 = run_phase_3_analysis(str(fixture_path))
        fixture_results["phase_3"] = phase3

        if phase3["status"] in ["✓", "⚠️"]:
            if "error" in phase3:
                print(f"      ⚠️  {phase3['error']}")
            else:
                print(f"      Audio peaks: {phase3['audio_peaks']} | Motion bursts: {phase3['motion_bursts']}")
                print(f"      Person frames: {phase3['person_frames']}/{phase3['total_frames_checked']}")
                print(f"      Fused: {phase3['fused']} → Filtered: {phase3['filtered']}")
                print(f"      Avg confidence: {phase3['avg_confidence']:.2f}")
                if phase3["filtered"] > 0:
                    print(f"      Dives: {phase3['dives']}")
                if phase3["filtered"] != expected_dives:
                    print(f"      ⚠️  Expected {expected_dives}, got {phase3['filtered']}")
        else:
            print(f"      ❌ Error: {phase3['error']}")

        results[fixture_name] = fixture_results

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    issues = []

    for fixture_name, expected_dives, desc in fixtures:
        if fixture_name not in results:
            continue

        res = results[fixture_name]

        # Check Phase 1
        p1_dives = res["phase_1"].get("filtered", 0) if "error" not in res["phase_1"] else None
        p2_dives = res["phase_2"].get("filtered", 0) if "error" not in res["phase_2"] else None
        p3_dives = res["phase_3"].get("filtered", 0) if "error" not in res["phase_3"] else None

        print(f"\n{fixture_name}:")
        print(f"  Expected: {expected_dives}")
        print(f"  Phase 1: {p1_dives if p1_dives is not None else 'ERROR'}")
        print(f"  Phase 2: {p2_dives if p2_dives is not None else 'ERROR'}")
        print(f"  Phase 3: {p3_dives if p3_dives is not None else 'ERROR'}")

        # Identify issues
        if fixture_name == "multi_dive.mp4" and p1_dives == 1 and expected_dives >= 3:
            issues.append({
                "file": fixture_name,
                "severity": "🔴 CRITICAL",
                "issue": "Merging bug: 37 peaks merged into 1 dive instead of 3+",
                "cause": "merge_overlapping_dives() logic needs fixing",
            })

        if fixture_name == "edge_cases/false_positive.mp4" and expected_dives == 0:
            if p1_dives and p1_dives > 0:
                issues.append({
                    "file": fixture_name,
                    "severity": "⚠️  HIGH",
                    "issue": f"False positive not rejected: Phase 1 found {p1_dives} dives",
                    "cause": "Phase 2/3 validation not rejecting loud noise",
                })

    print("\n" + "=" * 90)
    print("IDENTIFIED ISSUES")
    print("=" * 90)

    if not issues:
        print("\n✅ No critical issues found!")
    else:
        for issue in issues:
            print(f"\n{issue['severity']} {issue['file']}")
            print(f"  Issue: {issue['issue']}")
            print(f"  Cause: {issue['cause']}")

    # Save results to JSON
    results_file = Path("/tmp/comprehensive_diagnostics.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n✓ Full results saved to: {results_file}")
    print("Use `cat /tmp/comprehensive_diagnostics.json | jq` to view structured results")


if __name__ == "__main__":
    main()
