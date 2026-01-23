#!/usr/bin/env python3
"""
Quick real-world test of audio splash detection on your diving videos.

Works with just numpy, scipy, av (no librosa needed).
"""

import sys
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy import signal
import soundfile as sf


def extract_audio_ffmpeg(video_path, sr=22050):
    """Extract audio from video using FFmpeg (simpler and more reliable)."""
    print(f"  Extracting audio from: {Path(video_path).name}")

    # Use temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        # Extract audio with FFmpeg
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-y",
            audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        # Load audio
        audio, sr_read = sf.read(audio_path)

        if sr_read != sr:
            print(f"    Warning: actual sample rate {sr_read} != requested {sr}")

        return audio.astype(np.float32), sr_read

    finally:
        # Clean up temp file
        if Path(audio_path).exists():
            try:
                os.remove(audio_path)
            except:
                pass


def detect_splash_peaks_simple(audio, sr=22050, threshold_db=-25.0, min_distance_sec=5.0):
    """
    Detect splash peaks in audio using scipy (no librosa needed).

    Returns: List of (timestamp, amplitude_db) tuples
    """
    print(f"  Analyzing audio (sample rate: {sr}Hz, duration: {len(audio)/sr:.1f}s)...")

    # Compute RMS energy in windows
    frame_length = 2048
    hop_length = 512  # ~23ms windows

    # Split audio into frames
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    rms = np.zeros(n_frames)

    for i in range(n_frames):
        frame = audio[i * hop_length : i * hop_length + frame_length]
        if len(frame) > 0:
            rms[i] = np.sqrt(np.mean(frame**2))

    # Convert to dB
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

    # Find peaks
    min_distance_frames = int(min_distance_sec * sr / hop_length)
    peaks, properties = signal.find_peaks(
        rms_db,
        height=threshold_db,
        distance=min_distance_frames,
        prominence=5,
    )

    # Convert frame indices to timestamps
    peak_times = (peaks * hop_length) / sr
    peak_amplitudes = rms_db[peaks]

    results = list(zip(peak_times, peak_amplitudes))
    results = sorted(results, key=lambda x: x[0])

    return results


def create_dive_events(peaks):
    """Create dive events from audio peaks."""
    dives = []
    for idx, (splash_time, amplitude) in enumerate(peaks, 1):
        # Normalize amplitude: assume -40dB to 0dB range
        normalized_amplitude = max(0.0, min(1.0, (amplitude + 40) / 40))
        confidence = 0.5 + (normalized_amplitude * 0.5)

        dive = {
            "dive_number": idx,
            "splash_time": splash_time,
            "start_time": max(0.0, splash_time - 10.0),
            "end_time": splash_time + 3.0,
            "confidence": confidence,
            "amplitude_db": amplitude,
        }
        dives.append(dive)

    return dives


def extract_clip_ffmpeg(video_path, start_time, end_time, output_path):
    """Extract clip using FFmpeg stream copy."""
    duration = end_time - start_time

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def extract_all_clips(video_path, dives, output_dir):
    """Extract all dive clips."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {len(dives)} clips with FFmpeg stream copy...")

    results = {}
    for dive in dives:
        filename = f"dive_{dive['dive_number']:03d}.mp4"
        output_path = output_dir / filename

        try:
            success = extract_clip_ffmpeg(
                video_path,
                dive["start_time"],
                dive["end_time"],
                str(output_path),
            )

            if success:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                results[dive["dive_number"]] = (True, output_path, size_mb)
            else:
                results[dive["dive_number"]] = (False, None, None)

        except Exception as e:
            results[dive["dive_number"]] = (False, None, str(e))

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_real_video.py video.MOV [threshold]")
        print(f"\nTesting with: IMG_6447.MOV (small, 1 dive)")
        video = "IMG_6447.MOV"
        threshold = -25.0
    else:
        video = sys.argv[1]
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else -25.0

    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"🧪 Testing: {Path(video).name}")
    print("=" * 80)

    start_total = time.time()

    try:
        # Step 1: Extract audio
        print("\n[1/4] 🔊 Extracting audio...")
        start = time.time()
        audio, sr = extract_audio_ffmpeg(video, sr=22050)
        elapsed = time.time() - start
        print(f"  ✓ Audio extracted ({elapsed:.1f}s, {len(audio)/sr:.1f}s duration)")

        # Step 2: Detect peaks
        print(f"\n[2/4] 🌊 Detecting splashes (threshold: {threshold}dB)...")
        start = time.time()
        peaks = detect_splash_peaks_simple(audio, sr=sr, threshold_db=threshold)
        elapsed = time.time() - start
        print(f"  ✓ Detection complete ({elapsed:.1f}s)")
        print(f"  Found {len(peaks)} potential splashes")

        if peaks:
            print("\n  Splash details:")
            for i, (time_sec, amp_db) in enumerate(peaks[:5], 1):
                conf_pct = max(0, min(100, (amp_db + 40) / 40 * 100))
                print(f"    {i}. {time_sec:7.2f}s @ {amp_db:6.1f}dB (confidence {conf_pct:5.1f}%)")
            if len(peaks) > 5:
                print(f"    ... and {len(peaks) - 5} more")

        if not peaks:
            print("\n  ⚠️  No splashes detected. Try lower threshold:")
            print(f"     python test_real_video.py {video} -30")
            return

        # Step 3: Create dive events
        print(f"\n[3/4] 🔗 Creating dive events...")
        dives = create_dive_events(peaks)
        print(f"  ✓ Created {len(dives)} dive events")

        # Step 4: Extract clips
        print(f"\n[4/4] ✂️  Extracting dive clips...")
        output_dir = "./test_output"
        results = extract_all_clips(video, dives, output_dir)

        success_count = sum(1 for s, _, _ in results.values() if s)
        print(f"  ✓ Successfully extracted {success_count}/{len(dives)} clips")

        print("\n  Extracted files:")
        for dive_num in sorted(results.keys()):
            success, path, size_mb = results[dive_num]
            if success:
                print(f"    ✓ dive_{dive_num:03d}.mp4 ({size_mb:.1f}MB)")
            else:
                print(f"    ✗ dive_{dive_num:03d}.mp4")

        total_time = time.time() - start_total

        # Summary
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETE")
        print("=" * 80)
        print(f"📹 Video: {Path(video).name}")
        print(f"⏱️  Total processing time: {total_time:.1f}s")
        print(f"🌊 Splashes detected: {len(peaks)}")
        print(f"✂️  Clips extracted: {success_count}")
        print(f"📁 Output folder: {output_dir}")

        if success_count > 0:
            total_size = sum(s for _, _, s in results.values() if s) or 0
            print(f"💾 Total output size: {total_size:.1f}MB")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
