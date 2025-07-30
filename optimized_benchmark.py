#!/usr/bin/env python3
"""
Advanced optimization benchmark for dive detection.
Tests multiple optimization strategies against the current implementation.
"""

import argparse
import time
import cv2
import threading
import queue
from collections import deque
import slAIcer
import numpy as np

def benchmark_current_threading(video_path, diver_zone, splash_zone, board_y, water_y, num_frames):
    """Benchmark current threading implementation"""
    print("🔬 Benchmarking current threading implementation...")

    start_time = time.time()
    frame_gen = slAIcer.frame_generator(video_path)

    # Process frames
    count = 0
    for idx, frame in frame_gen:
        if count >= num_frames:
            break
        count += 1

    end_time = time.time()
    elapsed = end_time - start_time
    fps = count / elapsed if elapsed > 0 else 0

    return elapsed, fps, count

def benchmark_adaptive_detection(video_path, diver_zone, splash_zone, board_y, water_y, num_frames):
    """Benchmark with adaptive MediaPipe detection based on state"""
    print("🎯 Benchmarking adaptive MediaPipe detection...")

    import mediapipe as mp
    from slAIcer import suppress_stderr, detect_splash_combined

    mp_pose = mp.solutions.pose

    start_time = time.time()
    frame_gen = slAIcer.frame_generator(video_path)

    # Initialize pose detector
    with suppress_stderr():
        pose = mp_pose.Pose(static_image_mode=True)

    # State machine
    WAITING = 0
    DIVER_ON_PLATFORM = 1
    DIVING = 2

    state = WAITING
    consecutive_diver_frames = 0
    consecutive_no_diver_frames = 0
    frames_without_diver = 0
    diver_detection_threshold = 3

    # Pose detection cooldown
    last_pose_detection_frame = -1
    pose_cooldown_frames = 6  # Skip ~0.2s at 30fps

    prev_gray = None
    count = 0

    try:
        for idx, frame in frame_gen:
            if count >= num_frames:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Always do splash detection (cheap)
            if splash_zone and prev_gray is not None:
                band_top = int(splash_zone[0] * h)
                band_bot = int(splash_zone[1] * h)
                splash_band = gray[band_top:band_bot, :]
                prev_band = prev_gray[band_top:band_bot, :]
                splash_this_frame, _, _ = detect_splash_combined(splash_band, prev_band, 12.0)

            prev_gray = gray

            # Adaptive pose detection based on state
            should_detect_pose = False
            diver_in_zone = False

            if state == WAITING:
                # Always detect in waiting state
                should_detect_pose = True
            elif state == DIVER_ON_PLATFORM:
                # Use cooldown when diver is stable on platform
                if idx - last_pose_detection_frame >= pose_cooldown_frames:
                    should_detect_pose = True
            elif state == DIVING:
                # Skip pose detection during diving (diver is out of bounds)
                # Only detect occasionally to check if diver reappears (shouldn't happen)
                if idx - last_pose_detection_frame >= pose_cooldown_frames * 3:
                    should_detect_pose = True

            # Perform pose detection if needed
            if should_detect_pose and diver_zone:
                left_norm, top_norm, right_norm, bottom_norm = diver_zone
                left = int(left_norm * w)
                top = int(top_norm * h)
                right = int(right_norm * w)
                bottom = int(bottom_norm * h)

                roi = rgb[top:bottom, left:right]
                if roi.size > 0:
                    res = pose.process(roi)
                    diver_in_zone = res.pose_landmarks is not None
                    last_pose_detection_frame = idx

            # Update counters
            if diver_in_zone:
                consecutive_diver_frames += 1
                consecutive_no_diver_frames = 0
            else:
                consecutive_diver_frames = 0
                consecutive_no_diver_frames += 1

            # State transitions
            if state == WAITING:
                if consecutive_diver_frames >= diver_detection_threshold:
                    state = DIVER_ON_PLATFORM
                    print(f"  Frame {idx}: State -> DIVER_ON_PLATFORM")
            elif state == DIVER_ON_PLATFORM:
                if consecutive_no_diver_frames >= diver_detection_threshold:
                    state = DIVING
                    frames_without_diver = consecutive_no_diver_frames
                    print(f"  Frame {idx}: State -> DIVING")
            elif state == DIVING:
                if not diver_in_zone:
                    frames_without_diver += 1
                else:
                    frames_without_diver = 0

                # Simplified end condition for benchmark
                if frames_without_diver >= 60:  # ~2 seconds
                    print(f"  Frame {idx}: Dive complete, resetting to WAITING")
                    state = WAITING
                    frames_without_diver = 0

            count += 1

    finally:
        pose.close()

    end_time = time.time()
    elapsed = end_time - start_time
    fps = count / elapsed if elapsed > 0 else 0

    return elapsed, fps, count

def benchmark_frame_skipping(video_path, diver_zone, splash_zone, board_y, water_y, num_frames):
    """Benchmark with intelligent frame skipping"""
    print("⚡ Benchmarking intelligent frame skipping...")

    import mediapipe as mp
    from slAIcer import suppress_stderr, detect_splash_combined

    mp_pose = mp.solutions.pose

    start_time = time.time()
    frame_gen = slAIcer.frame_generator(video_path)

    # Initialize pose detector
    with suppress_stderr():
        pose = mp_pose.Pose(static_image_mode=True)

    # Skip frames when nothing interesting is happening
    frame_skip_factor = 1  # Process every Nth frame
    last_interesting_frame = 0
    skip_threshold = 30  # If nothing interesting for 30 frames, start skipping

    prev_gray = None
    count = 0
    processed_count = 0

    try:
        for idx, frame in frame_gen:
            if count >= num_frames:
                break

            # Decide whether to process this frame
            should_process = (idx % frame_skip_factor == 0) or (idx - last_interesting_frame < skip_threshold)

            if should_process:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Splash detection
                interesting = False
                if splash_zone and prev_gray is not None:
                    band_top = int(splash_zone[0] * h)
                    band_bot = int(splash_zone[1] * h)
                    splash_band = gray[band_top:band_bot, :]
                    prev_band = prev_gray[band_top:band_bot, :]
                    splash_this_frame, score, _ = detect_splash_combined(splash_band, prev_band, 12.0)
                    if splash_this_frame or score > 5.0:
                        interesting = True

                # Pose detection
                if diver_zone:
                    left_norm, top_norm, right_norm, bottom_norm = diver_zone
                    left = int(left_norm * w)
                    top = int(top_norm * h)
                    right = int(right_norm * w)
                    bottom = int(bottom_norm * h)

                    roi = rgb[top:bottom, left:right]
                    if roi.size > 0:
                        res = pose.process(roi)
                        if res.pose_landmarks:
                            interesting = True

                if interesting:
                    last_interesting_frame = idx
                    frame_skip_factor = 1  # Process every frame when interesting
                else:
                    frame_skip_factor = min(3, frame_skip_factor + 1)  # Skip up to every 3rd frame

                prev_gray = gray
                processed_count += 1

            count += 1

    finally:
        pose.close()

    end_time = time.time()
    elapsed = end_time - start_time
    fps = count / elapsed if elapsed > 0 else 0

    print(f"  Processed {processed_count}/{count} frames ({processed_count/count*100:.1f}%)")
    return elapsed, fps, count

def run_optimization_benchmark(video_path, num_frames=1000):
    """Run comprehensive optimization benchmark"""

    print(f"🚀 Advanced Optimization Benchmark")
    print(f"Video: {video_path}")
    print(f"Testing with {num_frames} frames")
    print("="*50)

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video info: {fps:.1f} fps, {total_frames} total frames")

    # Configure zones interactively (simplified for benchmark)
    # You can modify these or make them interactive
    diver_zone = (0.019, 0.486, 0.206, 0.669)  # From your last test
    splash_zone = (0.719, 0.788)  # From your last test
    board_y = 0.598
    water_y = 0.950

    print(f"Using zones: diver={diver_zone}, splash={splash_zone}")
    print("")

    results = {}

    # Benchmark 1: Current implementation baseline
    print("📊 Test 1: Current Threading Implementation")
    try:
        elapsed, achieved_fps, frames = benchmark_current_threading(
            video_path, diver_zone, splash_zone, board_y, water_y, num_frames)
        results['current'] = (elapsed, achieved_fps, frames)
        print(f"  Result: {elapsed:.2f}s, {achieved_fps:.1f} fps")
    except Exception as e:
        print(f"  Error: {e}")
        results['current'] = (0, 0, 0)

    print("")

    # Benchmark 2: Adaptive MediaPipe detection
    print("📊 Test 2: Adaptive MediaPipe Detection")
    try:
        elapsed, achieved_fps, frames = benchmark_adaptive_detection(
            video_path, diver_zone, splash_zone, board_y, water_y, num_frames)
        results['adaptive'] = (elapsed, achieved_fps, frames)
        print(f"  Result: {elapsed:.2f}s, {achieved_fps:.1f} fps")
    except Exception as e:
        print(f"  Error: {e}")
        results['adaptive'] = (0, 0, 0)

    print("")

    # Benchmark 3: Intelligent frame skipping
    print("📊 Test 3: Intelligent Frame Skipping")
    try:
        elapsed, achieved_fps, frames = benchmark_frame_skipping(
            video_path, diver_zone, splash_zone, board_y, water_y, num_frames)
        results['skipping'] = (elapsed, achieved_fps, frames)
        print(f"  Result: {elapsed:.2f}s, {achieved_fps:.1f} fps")
    except Exception as e:
        print(f"  Error: {e}")
        results['skipping'] = (0, 0, 0)

    print("")

    # Results comparison
    print("🎯 Optimization Results Comparison:")
    print("-" * 60)

    baseline_time = results['current'][0] if results['current'][0] > 0 else 1
    baseline_fps = results['current'][1]

    for name, (elapsed, achieved_fps, frames) in results.items():
        if elapsed > 0:
            speedup = baseline_time / elapsed
            fps_improvement = achieved_fps / baseline_fps if baseline_fps > 0 else 0
            target_fps = fps
            realtime_ratio = achieved_fps / target_fps

            print(f"{name.capitalize():15s}: {elapsed:6.2f}s | {achieved_fps:6.1f} fps | "
                  f"{speedup:4.2f}x faster | {realtime_ratio:4.2f}x realtime")

    print("-" * 60)

    # Find best optimization
    best_name = max(results.keys(), key=lambda k: results[k][1] if results[k][1] > 0 else 0)
    best_fps = results[best_name][1]

    if best_fps > baseline_fps:
        improvement = best_fps / baseline_fps
        print(f"🏆 Best optimization: {best_name} ({improvement:.2f}x faster)")
        print(f"💡 Recommended for production use!")
    else:
        print("📊 Current implementation is already optimal for this test case.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced optimization benchmark for dive detection")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--frames", type=int, default=1000, help="Number of frames to test (default: 1000)")

    args = parser.parse_args()

    run_optimization_benchmark(args.video_path, args.frames)
