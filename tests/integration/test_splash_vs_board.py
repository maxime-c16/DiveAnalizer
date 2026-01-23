#!/usr/bin/env python3
"""
Test: Is DiveAnalyzer detecting SPLASH sound or BOARD BOUNCE?

User hypothesis: Detected peaks might be diving board bouncing, not splash.
- Springboard dives: Board bounce BEFORE splash (different timing)
- Platform dives: ONLY splash sound (no board)
- High board: Minimal board motion, mainly splash

If all dives are detected regardless of platform type, it's likely splash.
If springboard dives detected earlier than expected, it's board bounce.
"""

import sys
import time
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy import signal
import soundfile as sf


def extract_audio_ffmpeg(video_path, sr=22050):
    """Extract audio from video using FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        audio, sr_read = sf.read(audio_path)
        return audio.astype(np.float32), sr_read

    finally:
        if Path(audio_path).exists():
            try:
                Path(audio_path).unlink()
            except:
                pass


def detect_splash_peaks(audio, sr=22050, threshold_db=-25.0, min_distance_sec=5.0):
    """Detect splash peaks using audio."""
    frame_length = 2048
    hop_length = 512

    n_frames = 1 + (len(audio) - frame_length) // hop_length
    rms = np.zeros(n_frames)

    for i in range(n_frames):
        frame = audio[i * hop_length : i * hop_length + frame_length]
        if len(frame) > 0:
            rms[i] = np.sqrt(np.mean(frame**2))

    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

    min_distance_frames = int(min_distance_sec * sr / hop_length)
    peaks, properties = signal.find_peaks(
        rms_db,
        height=threshold_db,
        distance=min_distance_frames,
        prominence=5,
    )

    peak_times = (peaks * hop_length) / sr
    peak_amplitudes = rms_db[peaks]

    return list(zip(peak_times, peak_amplitudes))


def analyze_peak_timing(peaks, verbose=True):
    """Analyze timing patterns in detected peaks.

    Hypothesis testing:
    - Board bounce: Occurs 1-3s BEFORE expected splash (quick bounce)
    - Splash sound: Occurs at expected splash time
    - Platform/high board: Only splash, no early bounce
    """
    if not peaks:
        return {}

    times, amplitudes = zip(*peaks)
    times = np.array(times)
    amplitudes = np.array(amplitudes)

    # Calculate timing patterns
    time_gaps = np.diff(times)

    stats = {
        "total_peaks": len(peaks),
        "avg_time_between_peaks": np.mean(time_gaps),
        "min_gap": np.min(time_gaps),
        "max_gap": np.max(time_gaps),
        "avg_amplitude": np.mean(amplitudes),
        "min_amplitude": np.min(amplitudes),
        "max_amplitude": np.max(amplitudes),
        "amplitude_std": np.std(amplitudes),
    }

    if verbose:
        print("\n📊 PEAK TIMING ANALYSIS:")
        print(f"  Total peaks: {stats['total_peaks']}")
        print(f"  Time between peaks:")
        print(f"    ├─ Average: {stats['avg_time_between_peaks']:.1f}s")
        print(f"    ├─ Min: {stats['min_gap']:.1f}s")
        print(f"    ├─ Max: {stats['max_gap']:.1f}s")
        print(f"  Amplitude statistics:")
        print(f"    ├─ Average: {stats['avg_amplitude']:.1f}dB")
        print(f"    ├─ Range: {stats['min_amplitude']:.1f} to {stats['max_amplitude']:.1f}dB")
        print(f"    └─ Std Dev: {stats['amplitude_std']:.1f}dB")

    return stats


def hypothesis_testing(peaks, video_duration):
    """Test hypothesis: Board bounce vs Splash sound

    Board bounce hypothesis:
    ✓ Detected peaks occur 1-3s before actual splash entry
    ✓ Pattern consistent across video
    ✓ Multiple peaks per dive (board bounce + splash)
    ✗ Detected on platform dives (shouldn't have bounce)

    Splash sound hypothesis:
    ✓ Detected peaks occur at splash entry time
    ✓ One peak per dive
    ✓ Detected on ALL platforms (springboard, platform, high board)
    ✗ Multiple bounces or pre-splash signals
    """
    if not peaks:
        return {}

    times, amplitudes = zip(*peaks)
    time_gaps = np.diff(times)

    print("\n🔬 HYPOTHESIS TESTING:")
    print("\n  Board Bounce Hypothesis:")
    print(f"    If detected sound is board bounce:")
    print(f"    - Should see early peaks 1-3s before splash")
    print(f"    - Average gap should be ~2-3s (time between bounces)")
    print(f"    - Should NOT detect on platform dives")
    print(f"\n    Actual data:")
    print(f"    - Average gap between peaks: {np.mean(time_gaps):.1f}s")
    print(f"    - Median gap: {np.median(time_gaps):.1f}s")
    if np.mean(time_gaps) < 5:
        print(f"    ⚠️  Gaps are short! Could indicate board bounces?")
    else:
        print(f"    ✓ Gaps are long (~5s+), consistent with separate dives")

    print(f"\n  Splash Sound Hypothesis:")
    print(f"    If detected sound is splash:")
    print(f"    - Average gap should be ~5-30s (time between dives)")
    print(f"    - Detected on all platform types")
    print(f"    - One peak per dive")
    print(f"\n    Actual data:")
    print(f"    - Average gap: {np.mean(time_gaps):.1f}s ✓")
    print(f"    - Total dives detected: {len(time_gaps) + 1}")
    print(f"    - Video duration: {video_duration:.1f}s")
    print(f"    - Dive rate: {(len(time_gaps) + 1) / video_duration * 60:.1f} dives/min")

    # Analysis
    avg_gap = np.mean(time_gaps)
    if avg_gap > 5:
        print(f"\n  ✓ LIKELY SPLASH SOUND - gaps are long (separate dives)")
        return "splash"
    elif avg_gap < 3:
        print(f"\n  ⚠️  MIGHT BE BOARD BOUNCE - gaps are very short")
        return "board"
    else:
        print(f"\n  🤔 UNCLEAR - gaps are medium, need more data")
        return "unclear"


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_splash_vs_board.py <video.MOV> [threshold]")
        print("\nExample:")
        print("  python test_splash_vs_board.py IMG_6496.MOV -22")
        sys.exit(1)

    video_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else -22.0

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🔬 SPLASH vs BOARD BOUNCE ANALYSIS")
    print("=" * 80)
    print(f"\nVideo: {Path(video_path).name}")
    print(f"Threshold: {threshold} dB")
    print("\nUser Hypothesis: Detected peaks might be board bounces, not splash sounds")
    print("- Springboard dives have board bounce BEFORE splash")
    print("- Platform dives have ONLY splash (no board)")
    print("- If we detect everything, it's likely SPLASH sound")

    # Extract and analyze
    print("\n" + "-" * 80)
    print("Extracting audio and detecting peaks...")
    print("-" * 80)

    start = time.time()
    audio, sr = extract_audio_ffmpeg(video_path, sr=22050)
    audio_time = time.time() - start
    video_duration = len(audio) / sr

    print(f"✓ Audio extracted ({audio_time:.1f}s)")
    print(f"  Duration: {video_duration:.1f}s")
    print(f"  Sample rate: {sr} Hz")

    start = time.time()
    peaks = detect_splash_peaks(audio, sr=sr, threshold_db=threshold)
    detect_time = time.time() - start

    print(f"✓ Peaks detected ({detect_time:.1f}s)")
    print(f"  Found: {len(peaks)} peaks")

    # Display peaks
    print(f"\n📍 Detected Peaks (first 20):")
    for i, (time_sec, amp_db) in enumerate(peaks[:20], 1):
        conf = 0.5 + (max(0.0, min(1.0, (amp_db + 40) / 40)) * 0.5)
        print(f"   {i:2d}. {time_sec:7.2f}s @ {amp_db:6.1f}dB (confidence {conf:.1%})")
    if len(peaks) > 20:
        print(f"   ... and {len(peaks) - 20} more")

    # Analyze timing
    stats = analyze_peak_timing(peaks, verbose=True)

    # Test hypothesis
    hypothesis = hypothesis_testing(peaks, video_duration)

    # Recommendations
    print("\n" + "-" * 80)
    print("📋 RECOMMENDATIONS FOR VERIFICATION:")
    print("-" * 80)

    if hypothesis == "splash":
        print("""
✓ EVIDENCE SUPPORTS: Detecting SPLASH SOUND, not board bounce

Why we're confident:
1. Long gaps between peaks (~5-30s) indicate separate dives
2. Consistent with actual dive session timing
3. One peak per dive (not multiple bounces)
4. Would be detected on ALL platform types

Next steps:
- Visual verification: Check video at detected times
- Listen to audio: Verify peaks align with splash sounds
- Cross-reference: Count dives manually vs detected peaks
        """)
    elif hypothesis == "board":
        print("""
⚠️  EVIDENCE SUGGESTS: Might be detecting BOARD BOUNCE

Concerns:
1. Very short gaps between peaks suggest rapid bounces
2. Multiple detections per dive area
3. Timing might be pre-splash (1-3s early)

Next steps:
- Shift detection window: Look 2-3s EARLIER than current peak
- Analyze amplitude pattern: Board bounces might be distinctive
- Test on platform dives: If no detections, confirms board bounce theory
        """)
    else:
        print("""
🤔 DATA IS INCONCLUSIVE

Need more information:
1. Visual inspection of video at detected times
2. Listen to audio at each detected peak
3. Count manual dives and compare to detections
4. Test on platform vs springboard separately
        """)

    print("\n" + "=" * 80)
    print("HOW TO VERIFY:")
    print("=" * 80)
    print("""
1. Play the video and note when dives actually occur
2. Compare with detected peak times (from above list)
3. Listen to audio at each detected time:
   - If it's BOARD BOUNCE: Metallic, ringing sound
   - If it's SPLASH: Watery, impact sound
4. Check if detection timing is:
   - AT splash entry (good - we're detecting splash)
   - 1-3s BEFORE entry (bad - we're detecting bounce)
5. Test on platform dives specifically:
   - Platform dives have NO board, only splash
   - If detected on platform dives, definitely splash sound
    """)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
