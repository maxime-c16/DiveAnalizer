#!/usr/bin/env python3
"""
Interactive splash detection testing with visual debugging
"""
import cv2
import numpy as np
import os
import sys

def detect_splash_frame_diff_improved(splash_band, prev_band, thresh_factor=2.0):
    """Improved frame difference with adaptive threshold"""
    if prev_band is None:
        return False, 0.0

    diff = cv2.absdiff(splash_band, prev_band)
    splash_score = np.mean(diff)

    # Adaptive threshold based on standard deviation
    std_dev = np.std(diff)
    adaptive_thresh = thresh_factor * std_dev + 8.0  # Higher base threshold

    return splash_score > adaptive_thresh, splash_score

def detect_splash_motion_intensity(splash_band, prev_band, intensity_thresh=15.0, coverage_thresh=0.15):
    """Motion-based detection focusing on intensity and coverage"""
    if prev_band is None:
        return False, 0.0

    diff = cv2.absdiff(splash_band, prev_band)

    # Apply threshold to get binary motion mask
    _, motion_mask = cv2.threshold(diff, intensity_thresh, 255, cv2.THRESH_BINARY)

    # Calculate motion coverage (percentage of pixels with significant motion)
    motion_pixels = np.sum(motion_mask > 0)
    total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
    motion_coverage = motion_pixels / total_pixels

    # Splash detected if high motion coverage
    is_splash = motion_coverage > coverage_thresh
    confidence = motion_coverage * 100  # Convert to percentage

    return is_splash, confidence

def detect_splash_temporal_consistency(splash_band, prev_band, history, consistency_frames=3):
    """Temporal consistency - splash must persist across multiple frames"""
    if prev_band is None:
        history.clear()
        return False, 0.0

    # Simple frame difference
    diff = cv2.absdiff(splash_band, prev_band)
    current_score = np.mean(diff)

    # Add to history
    history.append(current_score > 8.0)  # Basic threshold
    if len(history) > consistency_frames:
        history.pop(0)

    # Require splash in at least 2 out of last 3 frames
    if len(history) >= consistency_frames:
        recent_splashes = sum(history)
        is_splash = recent_splashes >= 2
        confidence = (recent_splashes / len(history)) * current_score
        return is_splash, confidence

    return False, current_score

def frame_generator(video_path, fps=30):
    """Yields (frame_index, frame) from a video file at a specific FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    interval = int(round(video_fps / fps))
    if interval == 0:
        interval = 1

    idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            yield saved_idx, frame
            saved_idx += 1
        idx += 1
    cap.release()

def interactive_splash_test(video_path, splash_zone_top=0.8, splash_zone_bottom=0.95):
    """Interactive test with visual debugging"""

    print(f"Testing splash detection with visual feedback on: {video_path}")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  ESC: Exit")
    print("  s: Save current frame")
    print("-" * 80)

    methods = {
        'Original': lambda sb, pb, h: (detect_splash_frame_diff_improved(sb, pb, 1.5), {}),
        'Motion': lambda sb, pb, h: (detect_splash_motion_intensity(sb, pb, 15.0, 0.15), {}),
        'Temporal': lambda sb, pb, h: (detect_splash_temporal_consistency(sb, pb, h), {})
    }

    prev_gray = None
    frame_count = 0
    paused = False
    temporal_history = []

    for idx, frame in frame_generator(video_path):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define splash zone
        band_top = int(splash_zone_top * h)
        band_bot = int(splash_zone_bottom * h)
        splash_band = gray[band_top:band_bot, :]

        # Create visualization frame
        vis_frame = frame.copy()

        # Draw splash zone
        cv2.rectangle(vis_frame, (0, band_top), (w, band_bot), (255, 0, 255), 2)
        cv2.putText(vis_frame, "Splash Detection Zone", (10, band_top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if prev_gray is not None:
            prev_band = prev_gray[band_top:band_bot, :]

            # Test each method
            y_offset = 30
            any_splash = False

            for method_name, method_func in methods.items():
                if method_name == 'Temporal':
                    (is_splash, confidence), _ = method_func(splash_band, prev_band, temporal_history)
                else:
                    (is_splash, confidence), _ = method_func(splash_band, prev_band, None)

                # Display results
                color = (0, 255, 0) if is_splash else (0, 0, 255)
                status = "SPLASH" if is_splash else "NO"
                text = f"{method_name}: {status} ({confidence:.2f})"
                cv2.putText(vis_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25

                if is_splash:
                    any_splash = True

            # Highlight splash zone if any method detects splash
            if any_splash:
                overlay = vis_frame.copy()
                cv2.rectangle(overlay, (0, band_top), (w, band_bot), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)

        # Frame info
        cv2.putText(vis_frame, f"Frame: {idx}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Splash Detection Test", vis_frame)

        if not paused:
            prev_gray = gray
            frame_count += 1

        # Handle key input
        key = cv2.waitKey(50 if not paused else 0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'} at frame {idx}")
        elif key == ord('s'):  # Save frame
            filename = f"splash_debug_frame_{idx}.jpg"
            cv2.imwrite(filename, vis_frame)
            print(f"Saved {filename}")

    cv2.destroyAllWindows()

def benchmark_methods(video_path, splash_zone_top=0.6, splash_zone_bottom=0.8, max_frames=20000):
    """Benchmark the improved methods"""

    print(f"Benchmarking improved splash detection methods")
    print(f"Video: {video_path}")
    print(f"Splash zone: {splash_zone_top:.1%} to {splash_zone_bottom:.1%}")
    print("-" * 80)

    methods = {
        'Original Frame Diff': lambda sb, pb, h: detect_splash_frame_diff_improved(sb, pb, 1.5),
        'Motion Intensity': lambda sb, pb, h: detect_splash_motion_intensity(sb, pb, 15.0, 0.15),
        'Temporal Consistency': lambda sb, pb, h: detect_splash_temporal_consistency(sb, pb, h)
    }

    results = {method: [] for method in methods.keys()}
    prev_gray = None
    frame_count = 0
    temporal_history = []

    for idx, frame in frame_generator(video_path):
        if frame_count >= max_frames:
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define splash zone
        band_top = int(splash_zone_top * h)
        band_bot = int(splash_zone_bottom * h)
        splash_band = gray[band_top:band_bot, :]

        if prev_gray is not None:
            prev_band = prev_gray[band_top:band_bot, :]

            # Test each method
            for method_name, method_func in methods.items():
                if method_name == 'Temporal Consistency':
                    is_splash, confidence = method_func(splash_band, prev_band, temporal_history)
                else:
                    is_splash, confidence = method_func(splash_band, prev_band, None)

                results[method_name].append((idx, is_splash, confidence))

                # Print significant detections
                if is_splash:
                    print(f"Frame {idx:3d}: {method_name:20s} -> SPLASH (confidence: {confidence:6.2f})")

        prev_gray = gray
        frame_count += 1

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)

    for method_name in methods.keys():
        detections = [r[1] for r in results[method_name]]
        total_detections = sum(detections)
        detection_rate = total_detections / len(detections) * 100 if detections else 0

        avg_confidence = np.mean([r[2] for r in results[method_name] if r[1]]) if total_detections > 0 else 0

        print(f"{method_name:20s}: {total_detections:3d} detections ({detection_rate:5.1f}%) | Avg confidence: {avg_confidence:6.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 improved_splash_test.py <video_path> [interactive|benchmark]")
        print("  Default mode: benchmark")
        sys.exit(1)

    video_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "benchmark"

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)

    if mode == "interactive":
        interactive_splash_test(video_path)
    else:
        benchmark_methods(video_path)
