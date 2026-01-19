#!/usr/bin/env python3
"""
Quick test script for Phase 1.

Run with: python scripts/quick_test.py video.mp4

Tests detection and extraction on your diving videos.
"""

import sys
import os
from pathlib import Path
import time
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from diveanalyzer import (
        extract_audio,
        detect_audio_peaks,
        fuse_signals_audio_only,
        extract_multiple_dives,
    )
    from diveanalyzer.extraction.ffmpeg import get_video_duration, format_time
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def test_video(
    video_path: str,
    output_dir: str = "./test_output",
    threshold: float = -25.0,
    confidence: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Test Phase 1 on a video file.

    Returns:
        Dictionary with results
    """
    video_path = str(Path(video_path).resolve())

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return {"success": False, "error": "File not found"}

    results = {
        "success": False,
        "video": video_path,
        "output_dir": output_dir,
        "dives_detected": 0,
        "dives_extracted": 0,
        "processing_time": 0,
        "errors": [],
    }

    start_time = time.time()

    try:
        # Get video info
        try:
            duration = get_video_duration(video_path)
            print(f"📹 Video: {Path(video_path).name}")
            print(f"⏱️  Duration: {format_time(duration)}")
        except Exception as e:
            print(f"⚠️  Could not get duration: {e}")
            duration = None

        # Step 1: Extract audio
        print("\n[1/4] 🔊 Extracting audio...")
        try:
            start = time.time()
            audio_path = extract_audio(video_path)
            elapsed = time.time() - start
            print(f"      ✓ Audio extracted ({elapsed:.1f}s)")
            if verbose:
                print(f"      File: {audio_path}")
        except Exception as e:
            error_msg = f"Failed to extract audio: {e}"
            print(f"      ✗ {error_msg}")
            results["errors"].append(error_msg)
            return results

        # Step 2: Detect splash peaks
        print(f"\n[2/4] 🌊 Detecting splashes (threshold: {threshold}dB)...")
        try:
            start = time.time()
            peaks = detect_audio_peaks(audio_path, threshold_db=threshold)
            elapsed = time.time() - start
            print(f"      ✓ Detection complete ({elapsed:.1f}s)")
            print(f"      Found {len(peaks)} potential splashes")

            if verbose and peaks:
                print("\n      Splash details:")
                for i, (time_sec, amp_db) in enumerate(peaks[:5], 1):
                    conf_pct = max(0, min(100, (amp_db + 40) / 40 * 100))
                    print(f"        {i}. {time_sec:7.2f}s @ {amp_db:6.1f}dB (confidence {conf_pct:5.1f}%)")
                if len(peaks) > 5:
                    print(f"        ... and {len(peaks) - 5} more")

            results["dives_detected"] = len(peaks)

        except Exception as e:
            error_msg = f"Failed to detect splashes: {e}"
            print(f"      ✗ {error_msg}")
            results["errors"].append(error_msg)
            return results

        if not peaks:
            print("\n⚠️  No splashes detected. Try lowering threshold:")
            print(f"    python scripts/quick_test.py {video_path} --threshold -30")
            results["success"] = True  # Not an error, just no dives found
            return results

        # Step 3: Create dive events
        print(f"\n[3/4] 🔗 Creating dive events...")
        try:
            dives = fuse_signals_audio_only(peaks)

            # Filter by confidence
            original_count = len(dives)
            dives_filtered = [d for d in dives if d.confidence >= confidence]
            filtered_count = len(dives_filtered)

            print(f"      ✓ Created {original_count} dive events")
            if filtered_count < original_count:
                print(f"      Filtered by confidence: {original_count} → {filtered_count}")
            else:
                dives_filtered = dives

            if verbose:
                print("\n      Dive events:")
                for dive in dives_filtered[:5]:
                    print(
                        f"        Dive #{dive.dive_number}: "
                        f"{dive.splash_time:7.2f}s "
                        f"({dive.start_time:.1f}s-{dive.end_time:.1f}s) "
                        f"confidence {dive.confidence:.1%}"
                    )
                if len(dives_filtered) > 5:
                    print(f"        ... and {len(dives_filtered) - 5} more")

            dives = dives_filtered

        except Exception as e:
            error_msg = f"Failed to create dive events: {e}"
            print(f"      ✗ {error_msg}")
            results["errors"].append(error_msg)
            return results

        # Step 4: Extract clips
        print(f"\n[4/4] ✂️  Extracting {len(dives)} dive clips...")

        if not dives:
            print("      (No dives to extract)")
            results["success"] = True
            return results

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            extraction_results = extract_multiple_dives(
                video_path,
                dives,
                output_path,
                verbose=verbose,
            )

            success_count = sum(1 for s, _, _ in extraction_results.values() if s)
            results["dives_extracted"] = success_count

            print(f"      ✓ Successfully extracted {success_count}/{len(dives)} clips")

            if verbose:
                print("\n      Extracted files:")
                for dive_num in sorted(extraction_results.keys()):
                    success, path, error = extraction_results[dive_num]
                    if success:
                        size = Path(path).stat().st_size / (1024 * 1024)
                        print(f"        ✓ dive_{dive_num:03d}.mp4 ({size:.1f}MB)")
                    else:
                        print(f"        ✗ dive_{dive_num:03d}.mp4 - {error}")

        except Exception as e:
            error_msg = f"Failed to extract clips: {e}"
            print(f"      ✗ {error_msg}")
            results["errors"].append(error_msg)
            return results

        results["success"] = True

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"\n❌ {error_msg}")
        results["errors"].append(error_msg)

    finally:
        results["processing_time"] = time.time() - start_time

    return results


def print_results(results: dict):
    """Print test results summary."""
    print_header("📊 Test Results Summary")

    if not results["success"] and results["errors"]:
        print(f"❌ Test FAILED\n")
        for error in results["errors"]:
            print(f"   • {error}")
        return

    print(f"✅ Test PASSED\n")
    print(f"📹 Video: {Path(results['video']).name}")
    print(f"⏱️  Processing time: {results['processing_time']:.1f}s")
    print(f"🌊 Splashes detected: {results['dives_detected']}")
    print(f"✂️  Clips extracted: {results['dives_extracted']}")
    print(f"📁 Output: {results['output_dir']}")

    if results["dives_extracted"] > 0:
        output_size = sum(
            f.stat().st_size for f in Path(results["output_dir"]).glob("*.mp4")
        ) / (1024 * 1024)
        print(f"💾 Total output size: {output_size:.1f}MB")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick test of DiveAnalyzer Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quick_test.py test_video.mp4
  python scripts/quick_test.py session.mp4 --threshold -20 --confidence 0.7
  python scripts/quick_test.py diving_clip.mp4 -o ./results -v
        """,
    )

    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "-o",
        "--output",
        default="./test_output",
        help="Output directory (default: ./test_output)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=-25.0,
        help="Audio threshold in dB (default: -25)",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Check imports
    if not IMPORTS_OK:
        print_header("❌ Installation Error")
        print(f"Failed to import DiveAnalyzer:\n{IMPORT_ERROR}\n")
        print("Fix with:")
        print("  pip install -r requirements.txt")
        print("  pip install -e .")
        sys.exit(1)

    # Run test
    print_header("🧪 DiveAnalyzer Phase 1 Quick Test")
    print(f"Video: {args.video}")
    print(f"Threshold: {args.threshold}dB")
    print(f"Confidence: {args.confidence}")

    results = test_video(
        args.video,
        output_dir=args.output,
        threshold=args.threshold,
        confidence=args.confidence,
        verbose=args.verbose,
    )

    # Print results
    print_results(results)

    # Return success code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
