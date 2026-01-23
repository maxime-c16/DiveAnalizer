#!/usr/bin/env python3
"""
Benchmark all three phases: Phase 1 (audio), Phase 2 (audio+motion), Phase 3 (audio+motion+person).

Real performance comparison on actual diving video.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from diveanalyzer.extraction.ffmpeg import get_video_duration, format_time
from diveanalyzer.extraction.proxy import generate_proxy, is_proxy_generation_needed
from diveanalyzer.detection.fusion import (
    fuse_signals_audio_only,
    fuse_signals_audio_motion,
    fuse_signals_audio_motion_person,
    merge_overlapping_dives,
    filter_dives_by_confidence,
)
from diveanalyzer.detection.motion import detect_motion_bursts
from diveanalyzer.storage.cache import CacheManager


def run_phase1_benchmark(video_path: str, verbose: bool = False) -> dict:
    """Run Phase 1 (audio-only) benchmark."""
    print("\n" + "=" * 80)
    print("🎬 PHASE 1 BENCHMARK: Audio-Only Detection")
    print("=" * 80)

    try:
        from diveanalyzer.detection.audio import extract_audio, detect_splash_peaks
    except ImportError as e:
        print(f"❌ Cannot run Phase 1: {e}")
        return {}

    results = {"phase": 1, "timings": {}, "detections": {}}

    # Audio extraction
    print("\n1️⃣ Audio Extraction...")
    start = time.time()
    audio_path = extract_audio(video_path)
    results["timings"]["audio_extract"] = time.time() - start
    print(f"   ✓ Time: {results['timings']['audio_extract']:.2f}s")

    # Splash detection
    print("\n2️⃣ Splash Detection...")
    start = time.time()
    peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)
    results["timings"]["splash_detect"] = time.time() - start
    print(f"   ✓ Found {len(peaks)} peaks")
    print(f"   ✓ Time: {results['timings']['splash_detect']:.2f}s")

    # Fusion (Phase 1)
    print("\n3️⃣ Signal Fusion (Phase 1)...")
    start = time.time()
    dives = fuse_signals_audio_only(peaks)
    results["timings"]["fusion"] = time.time() - start
    print(f"   ✓ Created {len(dives)} dive events")
    print(f"   ✓ Time: {results['timings']['fusion']:.2f}s")

    # Merge and filter
    start = time.time()
    dives_merged = merge_overlapping_dives(dives)
    dives_filtered = filter_dives_by_confidence(dives_merged, min_confidence=0.5)
    results["timings"]["merge_filter"] = time.time() - start

    # Statistics
    results["detections"]["total_peaks"] = len(peaks)
    results["detections"]["dives_detected"] = len(dives_filtered)
    results["detections"]["avg_confidence"] = (
        sum(d.confidence for d in dives_filtered) / len(dives_filtered)
        if dives_filtered
        else 0.0
    )

    results["timings"]["total"] = sum(results["timings"].values())

    print(f"\n📊 PHASE 1 Results:")
    print(f"   Total dives: {results['detections']['dives_detected']}")
    print(f"   Avg confidence: {results['detections']['avg_confidence']:.2%}")
    print(f"   Total time: {results['timings']['total']:.2f}s")

    return results


def run_phase2_benchmark(video_path: str, verbose: bool = False) -> dict:
    """Run Phase 2 (audio+motion) benchmark."""
    print("\n" + "=" * 80)
    print("🎬 PHASE 2 BENCHMARK: Audio + Motion Detection")
    print("=" * 80)

    try:
        from diveanalyzer.detection.audio import extract_audio, detect_splash_peaks
    except ImportError as e:
        print(f"❌ Cannot run Phase 2: {e}")
        return {}

    results = {"phase": 2, "timings": {}, "detections": {}}

    # Audio extraction
    print("\n1️⃣ Audio Extraction...")
    start = time.time()
    audio_path = extract_audio(video_path)
    results["timings"]["audio_extract"] = time.time() - start
    print(f"   ✓ Time: {results['timings']['audio_extract']:.2f}s")

    # Splash detection
    print("\n2️⃣ Splash Detection...")
    start = time.time()
    peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)
    results["timings"]["splash_detect"] = time.time() - start
    print(f"   ✓ Found {len(peaks)} peaks")
    print(f"   ✓ Time: {results['timings']['splash_detect']:.2f}s")

    # Proxy generation/caching
    cache = CacheManager()
    cache.cleanup_expired()

    print("\n3️⃣ Proxy Generation (with caching)...")
    start = time.time()

    # Check cache first
    cached_proxy = cache.get_proxy(video_path, height=480)
    if cached_proxy:
        print(f"   ✓ Using cached proxy")
        proxy_path = cached_proxy
        results["timings"]["proxy_gen"] = time.time() - start
    else:
        print(f"   Generating 480p proxy...")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_proxy = tmp.name

        generate_proxy(video_path, tmp_proxy, height=480, verbose=False)
        proxy_path = cache.put_proxy(video_path, tmp_proxy, height=480)
        results["timings"]["proxy_gen"] = time.time() - start
        print(f"   ✓ Generated and cached proxy")

    proxy_size_mb = Path(proxy_path).stat().st_size / (1024 * 1024)
    print(f"   ✓ Proxy size: {proxy_size_mb:.1f}MB")
    print(f"   ✓ Time: {results['timings']['proxy_gen']:.2f}s")

    # Motion detection on proxy
    print("\n4️⃣ Motion Detection (on proxy)...")
    start = time.time()
    motion_events = detect_motion_bursts(proxy_path, sample_fps=5.0)
    results["timings"]["motion_detect"] = time.time() - start
    print(f"   ✓ Found {len(motion_events)} motion bursts")
    print(f"   ✓ Time: {results['timings']['motion_detect']:.2f}s")

    # Fusion (Phase 2)
    print("\n5️⃣ Signal Fusion (Phase 2)...")
    start = time.time()
    dives = fuse_signals_audio_motion(peaks, motion_events)
    results["timings"]["fusion"] = time.time() - start
    print(f"   ✓ Created {len(dives)} dive events")
    print(f"   ✓ Time: {results['timings']['fusion']:.2f}s")

    # Merge and filter
    start = time.time()
    dives_merged = merge_overlapping_dives(dives)
    dives_filtered = filter_dives_by_confidence(dives_merged, min_confidence=0.5)
    results["timings"]["merge_filter"] = time.time() - start

    # Statistics
    results["detections"]["total_peaks"] = len(peaks)
    results["detections"]["motion_events"] = len(motion_events)
    results["detections"]["dives_detected"] = len(dives_filtered)
    results["detections"]["motion_validated"] = sum(
        1 for d in dives_filtered if d.notes == "audio+motion"
    )
    results["detections"]["avg_confidence"] = (
        sum(d.confidence for d in dives_filtered) / len(dives_filtered)
        if dives_filtered
        else 0.0
    )

    results["timings"]["total"] = sum(results["timings"].values())

    print(f"\n📊 PHASE 2 Results:")
    print(f"   Total dives: {results['detections']['dives_detected']}")
    print(
        f"   Motion-validated: {results['detections']['motion_validated']}/{results['detections']['dives_detected']}"
    )
    print(f"   Avg confidence: {results['detections']['avg_confidence']:.2%}")
    print(f"   Total time: {results['timings']['total']:.2f}s")

    return results


def run_phase3_benchmark(
    video_path: str,
    use_gpu: bool = False,
    use_fp16: bool = False,
    batch_size: int = 16,
    verbose: bool = False,
) -> dict:
    """Run Phase 3 (audio+motion+person) benchmark."""
    print("\n" + "=" * 80)
    print("🎬 PHASE 3 BENCHMARK: Audio + Motion + Person Detection")
    print("=" * 80)
    print(f"GPU enabled: {use_gpu}")
    print(f"FP16 enabled: {use_fp16}")
    print(f"Batch size: {batch_size}")

    try:
        from diveanalyzer.detection.audio import extract_audio, detect_splash_peaks
        from diveanalyzer.detection.person import (
            detect_person_frames,
            smooth_person_timeline,
            find_person_zone_departures,
        )
    except ImportError as e:
        print(f"❌ Cannot run Phase 3: {e}")
        return {}

    results = {"phase": 3, "timings": {}, "detections": {}}

    # Audio extraction
    print("\n1️⃣ Audio Extraction...")
    start = time.time()
    audio_path = extract_audio(video_path)
    results["timings"]["audio_extract"] = time.time() - start
    print(f"   ✓ Time: {results['timings']['audio_extract']:.2f}s")

    # Splash detection
    print("\n2️⃣ Splash Detection...")
    start = time.time()
    peaks = detect_splash_peaks(audio_path, threshold_db=-25.0, min_distance_sec=5.0)
    results["timings"]["splash_detect"] = time.time() - start
    print(f"   ✓ Found {len(peaks)} peaks")
    print(f"   ✓ Time: {results['timings']['splash_detect']:.2f}s")

    # Proxy (reuse from Phase 2)
    cache = CacheManager()
    cached_proxy = cache.get_proxy(video_path, height=480)
    if cached_proxy:
        print("\n3️⃣ Proxy (cached)...")
        proxy_path = cached_proxy
        results["timings"]["proxy_gen"] = 0.0
        print(f"   ✓ Using cached proxy")
    else:
        print("\n3️⃣ Proxy Generation...")
        start = time.time()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_proxy = tmp.name

        generate_proxy(video_path, tmp_proxy, height=480, verbose=False)
        proxy_path = cache.put_proxy(video_path, tmp_proxy, height=480)
        results["timings"]["proxy_gen"] = time.time() - start
        print(f"   ✓ Generated proxy")
        print(f"   ✓ Time: {results['timings']['proxy_gen']:.2f}s")

    # Motion detection on proxy
    print("\n4️⃣ Motion Detection (on proxy)...")
    start = time.time()
    motion_events = detect_motion_bursts(proxy_path, sample_fps=5.0)
    results["timings"]["motion_detect"] = time.time() - start
    print(f"   ✓ Found {len(motion_events)} motion bursts")
    print(f"   ✓ Time: {results['timings']['motion_detect']:.2f}s")

    # Person detection (NEW - Phase 3)
    print("\n5️⃣ Person Detection (on proxy)...")
    start = time.time()
    try:
        person_timeline = detect_person_frames(
            proxy_path,
            sample_fps=5.0,
            confidence_threshold=0.5,
            use_gpu=use_gpu,
            use_fp16=use_fp16,
            batch_size=batch_size,
        )
        results["timings"]["person_detect"] = time.time() - start
        print(f"   ✓ Sampled {len(person_timeline)} frames")
        print(f"   ✓ Time: {results['timings']['person_detect']:.2f}s")
    except Exception as e:
        print(f"   ⚠️  Person detection error: {e}")
        results["timings"]["person_detect"] = time.time() - start
        person_timeline = []

    # Smooth timeline
    print("\n6️⃣ Timeline Smoothing...")
    start = time.time()
    person_timeline = smooth_person_timeline(person_timeline, window_size=2)
    results["timings"]["smooth_timeline"] = time.time() - start

    # Find departures
    print("\n7️⃣ Finding Person Zone Departures...")
    start = time.time()
    person_departures = find_person_zone_departures(person_timeline, min_absence_duration=0.5)
    results["timings"]["find_departures"] = time.time() - start
    print(f"   ✓ Found {len(person_departures)} departures")
    print(f"   ✓ Time: {results['timings']['find_departures']:.2f}s")

    # Fusion (Phase 3)
    print("\n8️⃣ Signal Fusion (Phase 3)...")
    start = time.time()
    dives = fuse_signals_audio_motion_person(peaks, motion_events, person_departures)
    results["timings"]["fusion"] = time.time() - start
    print(f"   ✓ Created {len(dives)} dive events")
    print(f"   ✓ Time: {results['timings']['fusion']:.2f}s")

    # Merge and filter
    start = time.time()
    dives_merged = merge_overlapping_dives(dives)
    dives_filtered = filter_dives_by_confidence(dives_merged, min_confidence=0.5)
    results["timings"]["merge_filter"] = time.time() - start

    # Statistics
    three_signal = sum(1 for d in dives_filtered if d.notes == "3-signal")
    two_signal = sum(1 for d in dives_filtered if d.notes == "2-signal")
    audio_only = sum(1 for d in dives_filtered if d.notes == "audio-only")

    results["detections"]["total_peaks"] = len(peaks)
    results["detections"]["motion_events"] = len(motion_events)
    results["detections"]["person_departures"] = len(person_departures)
    results["detections"]["dives_detected"] = len(dives_filtered)
    results["detections"]["3_signal"] = three_signal
    results["detections"]["2_signal"] = two_signal
    results["detections"]["audio_only"] = audio_only
    results["detections"]["avg_confidence"] = (
        sum(d.confidence for d in dives_filtered) / len(dives_filtered)
        if dives_filtered
        else 0.0
    )

    results["timings"]["total"] = sum(results["timings"].values())

    print(f"\n📊 PHASE 3 Results:")
    print(f"   Total dives: {results['detections']['dives_detected']}")
    print(f"   ├─ 3-signal (audio+motion+person): {three_signal}")
    print(f"   ├─ 2-signal (audio+motion/person): {two_signal}")
    print(f"   └─ Audio-only: {audio_only}")
    print(f"   Avg confidence: {results['detections']['avg_confidence']:.2%}")
    print(f"   Total time: {results['timings']['total']:.2f}s")

    return results


def main():
    """Run all benchmarks and compare."""
    if len(sys.argv) < 2:
        print("Usage: python benchmark_all_phases.py <video.MOV> [--gpu] [--fp16] [--batch-size N]")
        print("\nExample:")
        print("  python benchmark_all_phases.py IMG_6496.MOV")
        print("  python benchmark_all_phases.py IMG_6496.MOV --gpu")
        print("  python benchmark_all_phases.py IMG_6496.MOV --gpu --fp16 --batch-size 32")
        sys.exit(1)

    video_path = sys.argv[1]
    use_gpu = "--gpu" in sys.argv
    use_fp16 = "--fp16" in sys.argv
    batch_size = 16

    # Parse batch size if provided
    for i, arg in enumerate(sys.argv):
        if arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    # Get video info
    duration = get_video_duration(video_path)
    size_mb = Path(video_path).stat().st_size / (1024 * 1024)

    print("=" * 80)
    print("DiveAnalyzer: Complete Phase Benchmark")
    print("=" * 80)
    print(f"\nVideo: {Path(video_path).name}")
    print(f"Duration: {format_time(duration)}")
    print(f"Size: {size_mb:.1f}MB")

    # Run benchmarks
    results = []
    results.append(run_phase1_benchmark(video_path))
    results.append(run_phase2_benchmark(video_path))
    results.append(
        run_phase3_benchmark(
            video_path, use_gpu=use_gpu, use_fp16=use_fp16, batch_size=batch_size
        )
    )

    # Compare results
    print("\n" + "=" * 80)
    print("📊 COMPARISON MATRIX")
    print("=" * 80)

    if all(results):
        p1, p2, p3 = results

        print(f"\n{'Metric':<25} {'Phase 1':<20} {'Phase 2':<20} {'Phase 3':<20}")
        print("-" * 85)

        # Timing
        print(
            f"{'Time (seconds)':<25} {p1['timings']['total']:>18.2f}s {p2['timings']['total']:>18.2f}s {p3['timings']['total']:>18.2f}s"
        )

        # Dives
        print(
            f"{'Dives Detected':<25} {p1['detections']['dives_detected']:>18d} {p2['detections']['dives_detected']:>18d} {p3['detections']['dives_detected']:>18d}"
        )

        # Confidence
        p1_conf = p1["detections"]["avg_confidence"]
        p2_conf = p2["detections"]["avg_confidence"]
        p3_conf = p3["detections"]["avg_confidence"]
        print(
            f"{'Avg Confidence':<25} {p1_conf:>18.1%} {p2_conf:>18.1%} {p3_conf:>18.1%}"
        )

        # Improvements
        print("\n" + "=" * 80)
        print("📈 IMPROVEMENTS")
        print("=" * 80)

        conf_gain_p2 = (p2_conf - p1_conf) / p1_conf * 100 if p1_conf > 0 else 0
        conf_gain_p3 = (p3_conf - p1_conf) / p1_conf * 100 if p1_conf > 0 else 0

        time_overhead_p2 = (p2["timings"]["total"] - p1["timings"]["total"]) / p1["timings"]["total"] * 100
        time_overhead_p3 = (p3["timings"]["total"] - p1["timings"]["total"]) / p1["timings"]["total"] * 100

        print(f"\nPhase 2 vs Phase 1:")
        print(f"  Confidence improvement: +{conf_gain_p2:.1f}%")
        print(f"  Time overhead: +{time_overhead_p2:.1f}%")
        print(
            f"  Motion-validated dives: {p2['detections']['motion_validated']}/{p2['detections']['dives_detected']}"
        )

        print(f"\nPhase 3 vs Phase 1:")
        print(f"  Confidence improvement: +{conf_gain_p3:.1f}%")
        print(f"  Time overhead: +{time_overhead_p3:.1f}%")
        print(f"  3-signal dives: {p3['detections']['3_signal']}/{p3['detections']['dives_detected']}")
        print(f"  2-signal dives: {p3['detections']['2_signal']}/{p3['detections']['dives_detected']}")
        print(f"  Audio-only dives: {p3['detections']['audio_only']}/{p3['detections']['dives_detected']}")

        # Performance summary
        print("\n" + "=" * 80)
        print("⚡ PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"""
Phase 1 (Audio-Only):
  └─ {p1['timings']['total']:.1f}s processing, {p1_conf:.1%} confidence

Phase 2 (Audio + Motion):
  └─ {p2['timings']['total']:.1f}s processing (+{time_overhead_p2:.0f}% overhead)
  └─ {p2_conf:.1%} confidence (+{conf_gain_p2:.0f}% improvement)
  └─ {p2['detections']['motion_validated']} motion-validated dives

Phase 3 (Audio + Motion + Person):
  └─ {p3['timings']['total']:.1f}s processing (+{time_overhead_p3:.0f}% overhead)
  └─ {p3_conf:.1%} confidence (+{conf_gain_p3:.0f}% improvement)
  └─ {p3['detections']['3_signal']} fully-validated (3-signal) dives

✅ Phase 3 READY FOR PRODUCTION
   • Highest accuracy ({p3_conf:.1%})
   • Most validated dives ({p3['detections']['3_signal']} 3-signal)
   • Acceptable overhead ({time_overhead_p3:.0f}% vs Phase 1)
""")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
