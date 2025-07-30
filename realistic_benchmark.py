#!/usr/bin/env python3
"""
Realistic optimization benchmark that actually performs pose detection work.
"""

import argparse
import time
import cv2
import threading
import queue
from collections import deque
import slAIcer
import numpy as np

def benchmark_current_implementation(video_path, diver_zone, splash_zone, board_y, water_y, num_frames):
    """Benchmark current find_next_dive implementation"""
    print("🔬 Benchmarking current find_next_dive implementation...")

    start_time = time.time()

    # Use actual find_next_dive function
    frame_gen = slAIcer.frame_generator(video_path)
    fps = slAIcer.get_video_fps(video_path)

    # Limit frames for benchmark
    limited_gen = ((idx, frame) for idx, frame in enumerate(frame_gen) if idx < num_frames)

    start_idx, end_idx, confidence = slAIcer.find_next_dive(
        limited_gen, board_y, water_y,
        splash_zone_top_norm=splash_zone[0], splash_zone_bottom_norm=splash_zone[1],
        diver_zone_norm=diver_zone, debug=False, use_threading=True, video_fps=fps
    )

    end_time = time.time()
    elapsed = end_time - start_time
    achieved_fps = num_frames / elapsed if elapsed > 0 else 0

    print(f"  Detected dive: start={start_idx}, end={end_idx}, confidence={confidence}")
    return elapsed, achieved_fps, num_frames

def benchmark_optimized_pose_detection(video_path, diver_zone, splash_zone, board_y, water_y, num_frames):
    """Benchmark with optimized pose detection strategy"""
    print("🎯 Benchmarking optimized pose detection...")

    import mediapipe as mp
    from slAIcer import suppress_stderr, detect_splash_combined

    mp_pose = mp.solutions.pose

    start_time = time.time()
    frame_gen = slAIcer.frame_generator(video_path)

    # Initialize pose detector
    with suppress_stderr():
        pose = mp_pose.Pose(static_image_mode=True)

    # Optimization parameters
    WAITING = 0
    DIVER_ON_PLATFORM = 1
    DIVING = 2

    state = WAITING
    consecutive_diver_frames = 0
    consecutive_no_diver_frames = 0
    frames_without_diver = 0
    diver_detection_threshold = 3

    # Pose detection optimization
    last_pose_detection_frame = -1
    pose_cooldown_frames = 6  # Skip ~0.2s at 30fps
    pose_detections_skipped = 0
    total_pose_opportunities = 0

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
            splash_this_frame = False
            if splash_zone and prev_gray is not None:
                band_top = int(splash_zone[0] * h)
                band_bot = int(splash_zone[1] * h)
                splash_band = gray[band_top:band_bot, :]
                prev_band = prev_gray[band_top:band_bot, :]
                splash_this_frame, _, _ = detect_splash_combined(splash_band, prev_band, 12.0)

            prev_gray = gray

            # Smart pose detection based on state
            should_detect_pose = False
            diver_in_zone = False

            total_pose_opportunities += 1

            if state == WAITING:
                # Always detect in waiting state
                should_detect_pose = True
            elif state == DIVER_ON_PLATFORM:
                # Use cooldown when diver is stable on platform
                frames_since_last = idx - last_pose_detection_frame
                if frames_since_last >= pose_cooldown_frames:
                    should_detect_pose = True
                else:
                    # Assume diver is still there during cooldown
                    diver_in_zone = True  # Optimistic assumption
            elif state == DIVING:
                # During diving, pose detection is mostly useless since diver left the zone
                # Only check occasionally to be safe
                frames_since_last = idx - last_pose_detection_frame
                if frames_since_last >= pose_cooldown_frames * 5:  # Much longer cooldown
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
            else:
                pose_detections_skipped += 1

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
                    start_idx = idx - consecutive_diver_frames + 1
            elif state == DIVER_ON_PLATFORM:
                if consecutive_no_diver_frames >= diver_detection_threshold:
                    state = DIVING
                    frames_without_diver = consecutive_no_diver_frames
            elif state == DIVING:
                if not diver_in_zone:
                    frames_without_diver += 1
                else:
                    frames_without_diver = 0

                # End condition check
                if splash_this_frame and frames_without_diver >= diver_detection_threshold:
                    end_idx = idx
                    break
                elif frames_without_diver >= 180:  # 6 seconds timeout
                    end_idx = idx
                    break

            count += 1

    finally:
        pose.close()

    end_time = time.time()
    elapsed = end_time - start_time
    fps = count / elapsed if elapsed > 0 else 0

    skip_percentage = (pose_detections_skipped / total_pose_opportunities) * 100
    print(f"  Skipped {pose_detections_skipped}/{total_pose_opportunities} pose detections ({skip_percentage:.1f}%)")

    return elapsed, fps, count

def benchmark_background_extraction(video_path, diver_zone, splash_zone, board_y, water_y, num_frames):
    """Benchmark with background dive extraction threads"""
    print("🚀 Benchmarking background dive extraction...")

    import mediapipe as mp
    from slAIcer import suppress_stderr, detect_splash_combined

    mp_pose = mp.solutions.pose

    # Simulate background extraction with threading
    extraction_queue = queue.Queue()
    extraction_results = []

    def background_extractor():
        """Simulate dive extraction in background"""
        while True:
            try:
                dive_data = extraction_queue.get(timeout=1.0)
                if dive_data is None:  # Shutdown signal
                    break

                # Simulate dive extraction work (normally would save video clips)
                start_idx, end_idx, video_path = dive_data
                time.sleep(0.1)  # Simulate extraction time
                extraction_results.append((start_idx, end_idx))
                extraction_queue.task_done()
            except queue.Empty:
                continue

    # Start background extraction thread
    extractor_thread = threading.Thread(target=background_extractor, daemon=True)
    extractor_thread.start()

    start_time = time.time()
    frame_gen = slAIcer.frame_generator(video_path)

    # Initialize pose detector
    with suppress_stderr():
        pose = mp_pose.Pose(static_image_mode=True)

    # Detection state
    WAITING = 0
    DIVER_ON_PLATFORM = 1
    DIVING = 2

    state = WAITING
    consecutive_diver_frames = 0
    consecutive_no_diver_frames = 0
    frames_without_diver = 0
    diver_detection_threshold = 3

    # Track found dives
    found_dives = []

    prev_gray = None
    count = 0

    try:
        for idx, frame in frame_gen:
            if count >= num_frames:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Splash detection
            splash_this_frame = False
            if splash_zone and prev_gray is not None:
                band_top = int(splash_zone[0] * h)
                band_bot = int(splash_zone[1] * h)
                splash_band = gray[band_top:band_bot, :]
                prev_band = prev_gray[band_top:band_bot, :]
                splash_this_frame, _, _ = detect_splash_combined(splash_band, prev_band, 12.0)

            prev_gray = gray

            # Pose detection (simplified for benchmark)
            diver_in_zone = False
            if diver_zone:
                left_norm, top_norm, right_norm, bottom_norm = diver_zone
                left = int(left_norm * w)
                top = int(top_norm * h)
                right = int(right_norm * w)
                bottom = int(bottom_norm * h)

                roi = rgb[top:bottom, left:right]
                if roi.size > 0:
                    res = pose.process(roi)
                    diver_in_zone = res.pose_landmarks is not None

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
                    start_idx = idx - consecutive_diver_frames + 1
            elif state == DIVER_ON_PLATFORM:
                if consecutive_no_diver_frames >= diver_detection_threshold:
                    state = DIVING
                    frames_without_diver = consecutive_no_diver_frames
            elif state == DIVING:
                if not diver_in_zone:
                    frames_without_diver += 1
                else:
                    frames_without_diver = 0

                # End condition check
                if splash_this_frame and frames_without_diver >= diver_detection_threshold:
                    end_idx = idx
                    # Immediately spawn background extraction
                    dive_data = (start_idx, end_idx, video_path)
                    extraction_queue.put(dive_data)
                    found_dives.append((start_idx, end_idx))

                    # Reset for next dive
                    state = WAITING
                    frames_without_diver = 0
                elif frames_without_diver >= 180:  # 6 seconds timeout
                    end_idx = idx
                    dive_data = (start_idx, end_idx, video_path)
                    extraction_queue.put(dive_data)
                    found_dives.append((start_idx, end_idx))

                    # Reset for next dive
                    state = WAITING
                    frames_without_diver = 0

            count += 1

    finally:
        # Cleanup
        pose.close()
        extraction_queue.put(None)  # Shutdown signal
        extractor_thread.join(timeout=2.0)

    end_time = time.time()
    elapsed = end_time - start_time
    fps = count / elapsed if elapsed > 0 else 0

    print(f"  Found {len(found_dives)} dives, {len(extraction_results)} extractions completed")

    return elapsed, fps, count

def run_realistic_benchmark(video_path, num_frames=500):
    """Run realistic optimization benchmark"""

    print(f"🚀 Realistic Optimization Benchmark")
    print(f"Video: {video_path}")
    print(f"Testing with {num_frames} frames")
    print("="*60)

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video info: {fps:.1f} fps, {total_frames} total frames")

    # Use predefined zones
    diver_zone = (0.019, 0.486, 0.206, 0.669)
    splash_zone = (0.719, 0.788)
    board_y = 0.598
    water_y = 0.950

    print(f"Using zones: diver={diver_zone}, splash={splash_zone}")
    print("")

    results = {}

    # Benchmark 1: Current implementation
    print("📊 Test 1: Current Implementation")
    try:
        elapsed, achieved_fps, frames = benchmark_current_implementation(
            video_path, diver_zone, splash_zone, board_y, water_y, num_frames)
        results['current'] = (elapsed, achieved_fps, frames)
        print(f"  Result: {elapsed:.2f}s, {achieved_fps:.1f} fps")
    except Exception as e:
        print(f"  Error: {e}")
        results['current'] = (0, 0, 0)

    print("")

    # Benchmark 2: Optimized pose detection
    print("📊 Test 2: Optimized Pose Detection")
    try:
        elapsed, achieved_fps, frames = benchmark_optimized_pose_detection(
            video_path, diver_zone, splash_zone, board_y, water_y, num_frames)
        results['optimized'] = (elapsed, achieved_fps, frames)
        print(f"  Result: {elapsed:.2f}s, {achieved_fps:.1f} fps")
    except Exception as e:
        print(f"  Error: {e}")
        results['optimized'] = (0, 0, 0)

    print("")

    # Benchmark 3: Background extraction
    print("📊 Test 3: Background Extraction")
    try:
        elapsed, achieved_fps, frames = benchmark_background_extraction(
            video_path, diver_zone, splash_zone, board_y, water_y, num_frames)
        results['background'] = (elapsed, achieved_fps, frames)
        print(f"  Result: {elapsed:.2f}s, {achieved_fps:.1f} fps")
    except Exception as e:
        print(f"  Error: {e}")
        results['background'] = (0, 0, 0)

    print("")

    # Results comparison
    print("🎯 Optimization Results Comparison:")
    print("-" * 80)

    baseline_time = results['current'][0] if results['current'][0] > 0 else 1
    baseline_fps = results['current'][1]

    for name, (elapsed, achieved_fps, frames) in results.items():
        if elapsed > 0:
            speedup = baseline_time / elapsed
            fps_improvement = achieved_fps / baseline_fps if baseline_fps > 0 else 0
            target_fps = fps
            realtime_ratio = achieved_fps / target_fps

            print(f"{name.capitalize():12s}: {elapsed:6.2f}s | {achieved_fps:6.1f} fps | "
                  f"{speedup:4.2f}x faster | {realtime_ratio:4.2f}x realtime")

    print("-" * 80)

    # Find best optimization
    valid_results = {k: v for k, v in results.items() if v[1] > 0}

    if valid_results:
        best_name = max(valid_results.keys(), key=lambda k: valid_results[k][1])
        best_fps = valid_results[best_name][1]

        if baseline_fps > 0 and best_fps > baseline_fps * 1.1:  # 10% improvement threshold
            improvement = best_fps / baseline_fps
            print(f"🏆 Best optimization: {best_name} ({improvement:.2f}x faster)")
            print(f"💡 Recommended for production use!")
        elif baseline_fps == 0:
            print(f"🏆 Best performing method: {best_name} ({best_fps:.1f} fps)")
            print(f"💡 Recommended since baseline failed!")
        else:
            print("📊 Optimizations show marginal improvement - current implementation is solid.")
    else:
        print("❌ No valid benchmark results obtained.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realistic optimization benchmark")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to test")

    args = parser.parse_args()

    run_realistic_benchmark(args.video_path, args.frames)
