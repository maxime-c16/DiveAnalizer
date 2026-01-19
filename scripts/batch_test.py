#!/usr/bin/env python3
"""
Batch test script for Phase 1.

Tests multiple videos and generates a report.

Run with: python scripts/batch_test.py [options]

Examples:
  python scripts/batch_test.py --folder ./test_videos
  python scripts/batch_test.py --pattern "dive_*.mp4"
  python scripts/batch_test.py --folder ./clips --threshold -24 --output ./results
"""

import sys
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.quick_test import test_video, print_header
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False


def find_videos(folder: Path, pattern: str = "*.mp4") -> List[Path]:
    """Find all video files matching pattern."""
    videos = []

    # Direct match
    if folder.is_file():
        return [folder]

    if folder.is_dir():
        videos.extend(folder.glob(pattern))
        videos.extend(folder.glob(pattern.replace("*.mp4", "*.mov")))
        videos.extend(folder.glob(pattern.replace("*.mp4", "*.avi")))

    return sorted(videos)


def batch_test(
    video_folders: List[Path],
    output_dir: Path,
    threshold: float = -25.0,
    confidence: float = 0.5,
    pattern: str = "*.mp4",
) -> Dict:
    """Test multiple videos and collect results."""
    results_all = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "threshold": threshold,
            "confidence": confidence,
            "pattern": pattern,
        },
        "summary": {
            "total_videos": 0,
            "successful": 0,
            "failed": 0,
            "total_dives_detected": 0,
            "total_dives_extracted": 0,
            "total_processing_time": 0,
            "total_output_size_mb": 0,
        },
        "videos": [],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all videos
    all_videos = []
    for folder in video_folders:
        all_videos.extend(find_videos(folder, pattern))

    print(f"\n📁 Found {len(all_videos)} video(s)")

    if not all_videos:
        print("❌ No videos found matching pattern")
        return results_all

    # Test each video
    for idx, video_path in enumerate(all_videos, 1):
        video_name = video_path.name
        video_output = output_dir / video_name.replace(".mp4", "")

        print(f"\n[{idx}/{len(all_videos)}] Testing: {video_name}")

        results = test_video(
            str(video_path),
            output_dir=str(video_output),
            threshold=threshold,
            confidence=confidence,
            verbose=False,
        )

        # Collect stats
        video_result = {
            "filename": video_name,
            "success": results["success"],
            "dives_detected": results["dives_detected"],
            "dives_extracted": results["dives_extracted"],
            "processing_time": results["processing_time"],
            "output_size_mb": 0,
        }

        if results["success"]:
            results_all["summary"]["successful"] += 1

            # Calculate output size
            try:
                output_size = sum(
                    f.stat().st_size
                    for f in Path(video_output).glob("*.mp4")
                    if f.is_file()
                ) / (1024 * 1024)
                video_result["output_size_mb"] = output_size
                results_all["summary"]["total_output_size_mb"] += output_size
            except:
                pass
        else:
            results_all["summary"]["failed"] += 1
            video_result["errors"] = results["errors"]

        results_all["summary"]["total_videos"] += 1
        results_all["summary"]["total_dives_detected"] += results["dives_detected"]
        results_all["summary"]["total_dives_extracted"] += results["dives_extracted"]
        results_all["summary"]["total_processing_time"] += results["processing_time"]

        results_all["videos"].append(video_result)

        # Print result
        if results["success"]:
            print(
                f"   ✓ {results['dives_detected']} detected, {results['dives_extracted']} extracted ({results['processing_time']:.1f}s)"
            )
        else:
            print(f"   ✗ Failed: {results['errors'][0] if results['errors'] else 'Unknown error'}")

    return results_all


def print_batch_summary(results: Dict):
    """Print batch test summary."""
    print_header("📊 Batch Test Summary")

    summary = results["summary"]

    print(f"Videos tested: {summary['total_videos']}")
    print(f"  ✓ Successful: {summary['successful']}")
    print(f"  ✗ Failed: {summary['failed']}")

    if summary["total_videos"] > 0:
        print(f"\n🌊 Total splashes detected: {summary['total_dives_detected']}")
        print(f"✂️  Total dives extracted: {summary['total_dives_extracted']}")
        print(f"⏱️  Total processing time: {summary['total_processing_time']:.1f}s")

        if summary["total_videos"] > 1:
            avg_time = summary["total_processing_time"] / summary["total_videos"]
            print(f"   Average per video: {avg_time:.1f}s")

        if summary["total_dives_extracted"] > 0:
            print(f"💾 Total output size: {summary['total_output_size_mb']:.1f}MB")

    print(f"\n📝 Details:")
    for video in results["videos"]:
        if video["success"]:
            print(
                f"   ✓ {video['filename']}: "
                f"{video['dives_detected']} detected, {video['dives_extracted']} extracted"
            )
        else:
            print(f"   ✗ {video['filename']}: Failed")


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch test DiveAnalyzer Phase 1 on multiple videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all MP4s in a folder
  python scripts/batch_test.py --folder ./test_videos

  # Test specific pattern
  python scripts/batch_test.py --folder ./clips --pattern "dive_*.mp4"

  # Test with custom parameters
  python scripts/batch_test.py --folder ./videos --threshold -24 --confidence 0.7

  # Save results to JSON
  python scripts/batch_test.py --folder ./videos --report results.json
        """,
    )

    parser.add_argument(
        "--folder",
        type=Path,
        action="append",
        help="Folder containing videos (can be used multiple times)",
    )
    parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="File pattern to match (default: *.mp4)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./batch_output"),
        help="Output directory for extracted clips (default: ./batch_output)",
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
        "-r",
        "--report",
        type=Path,
        help="Save results to JSON report",
    )

    args = parser.parse_args()

    # Check imports
    if not IMPORTS_OK:
        print("❌ Failed to import test utilities\n")
        print("Fix with:")
        print("  pip install -r requirements.txt")
        print("  pip install -e .")
        sys.exit(1)

    # Default to current directory if no folder specified
    if not args.folder:
        args.folder = [Path.cwd()]

    # Verify folders exist
    folders = []
    for folder in args.folder:
        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            continue
        folders.append(folder)

    if not folders:
        print("❌ No valid folders found")
        sys.exit(1)

    print_header("🧪 DiveAnalyzer Batch Test")
    print(f"Folders: {', '.join(str(f) for f in folders)}")
    print(f"Pattern: {args.pattern}")
    print(f"Threshold: {args.threshold}dB")
    print(f"Confidence: {args.confidence}")

    # Run batch test
    results = batch_test(
        folders,
        args.output,
        threshold=args.threshold,
        confidence=args.confidence,
        pattern=args.pattern,
    )

    # Print summary
    print_batch_summary(results)

    # Save report
    if args.report:
        save_results(results, args.report)

    # Return success/failure
    success = results["summary"]["failed"] == 0
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
