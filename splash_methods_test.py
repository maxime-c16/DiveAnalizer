#!/usr/bin/env python3
"""
Test script to compare different splash detection methods
"""
import cv2
import numpy as np
import os
import sys

def detect_splash_frame_diff(splash_band, prev_band, thresh=5.0):
    """Current method: Simple frame difference"""
    if prev_band is None:
        return False, 0.0

    diff = cv2.absdiff(splash_band, prev_band)
    splash_score = np.mean(diff)
    return splash_score > thresh, splash_score

def detect_splash_optical_flow(splash_band, prev_band, flow_thresh=2.0, magnitude_thresh=10.0):
    """Optical flow based splash detection"""
    if prev_band is None:
        return False, 0.0

    try:
        # Create simple grid of points for optical flow
        h, w = splash_band.shape
        y, x = np.mgrid[10:h-10:20, 10:w-10:20].reshape(2, -1).astype(np.float32)
        points = np.column_stack([x, y]).reshape(-1, 1, 2)

        if len(points) > 5:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_band, splash_band, points, None,
                winSize=(15, 15), maxLevel=2
            )

            if next_points is not None and status is not None:
                # Calculate movement magnitude
                good_points = next_points[status == 1]
                orig_points = points[status == 1]

                if len(good_points) > 5:
                    movement = np.linalg.norm(good_points - orig_points, axis=2)
                    mean_movement = np.mean(movement)
                    flow_splash = mean_movement > flow_thresh
                    return flow_splash, mean_movement
    except:
        pass

    return False, 0.0

def detect_splash_contour_analysis(splash_band, prev_band, area_thresh=500, contour_thresh=3):
    """Contour and foam analysis based splash detection"""
    if prev_band is None:
        return False, 0.0

    try:
        # Convert to HSV for better water/foam separation
        splash_bgr = cv2.cvtColor(cv2.merge([splash_band, splash_band, splash_band]), cv2.COLOR_GRAY2BGR)
        splash_hsv = cv2.cvtColor(splash_bgr, cv2.COLOR_BGR2HSV)

        # Detect white/bright areas (foam)
        lower_foam = np.array([0, 0, 180])  # Low saturation, high brightness
        upper_foam = np.array([180, 60, 255])
        foam_mask = cv2.inRange(splash_hsv, lower_foam, upper_foam)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foam_mask = cv2.morphologyEx(foam_mask, cv2.MORPH_CLOSE, kernel)
        foam_mask = cv2.morphologyEx(foam_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(foam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, 0.0

        # Analyze contours for splash characteristics
        total_foam_area = 0
        significant_contours = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_thresh:
                # Check if contour is irregular (characteristic of splash)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Splashes are typically irregular (low circularity)
                    if circularity < 0.7:
                        total_foam_area += area
                        significant_contours += 1

        # Splash detected if enough foam area and irregular contours
        is_splash = significant_contours >= contour_thresh and total_foam_area > area_thresh * 2
        confidence = total_foam_area / (splash_band.shape[0] * splash_band.shape[1])

        return is_splash, confidence
    except:
        return False, 0.0

def detect_splash_combined(splash_band, prev_band, splash_thresh=5.0):
    """Combined splash detection using multiple methods with voting"""
    # Test all three methods
    diff_splash, diff_score = detect_splash_frame_diff(splash_band, prev_band, splash_thresh)
    flow_splash, flow_score = detect_splash_optical_flow(splash_band, prev_band)
    contour_splash, contour_score = detect_splash_contour_analysis(splash_band, prev_band)

    # Voting system: at least 2 out of 3 methods must agree
    votes = sum([diff_splash, flow_splash, contour_splash])
    combined_confidence = (diff_score + flow_score * 10 + contour_score * 1000) / 3

    return votes >= 2, combined_confidence, {
        'frame_diff': (diff_splash, diff_score),
        'optical_flow': (flow_splash, flow_score),
        'contour': (contour_splash, contour_score),
        'votes': votes
    }

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

def test_splash_methods(video_path, splash_zone_top=0.8, splash_zone_bottom=0.95, max_frames=200):
    """Test all splash detection methods side by side"""

    methods = {
        'frame_diff': detect_splash_frame_diff,
        'optical_flow': detect_splash_optical_flow,
        'contour': detect_splash_contour_analysis,
        'combined': detect_splash_combined
    }

    results = {method: [] for method in methods.keys()}
    results['combined_details'] = []

    prev_gray = None
    frame_count = 0

    print(f"Testing splash detection methods on: {video_path}")
    print(f"Splash zone: {splash_zone_top:.1%} to {splash_zone_bottom:.1%} of frame height")
    print("-" * 80)

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
            for method_name in ['frame_diff', 'optical_flow', 'contour']:
                method_func = methods[method_name]
                is_splash, confidence = method_func(splash_band, prev_band)
                results[method_name].append((idx, is_splash, confidence))

            # Test combined method
            is_splash, confidence, details = methods['combined'](splash_band, prev_band)
            results['combined'].append((idx, is_splash, confidence))
            results['combined_details'].append((idx, details))

            # Print results for this frame if any method detects splash
            any_splash = any([results[method][-1][1] for method in ['frame_diff', 'optical_flow', 'contour', 'combined']])

            if any_splash:
                print(f"Frame {idx:3d}: ", end="")
                for method in ['frame_diff', 'optical_flow', 'contour']:
                    is_splash, conf = results[method][-1][1], results[method][-1][2]
                    status = "SPLASH" if is_splash else "   no "
                    print(f"{method:12s}: {status} ({conf:6.2f}) | ", end="")

                # Combined method details
                is_splash, conf = results['combined'][-1][1], results['combined'][-1][2]
                details = results['combined_details'][-1][1]
                votes = details['votes']
                print(f"Combined: {'SPLASH' if is_splash else '   no '} (votes:{votes}, conf:{conf:6.2f})")

        prev_gray = gray
        frame_count += 1

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)

    for method_name in methods.keys():
        if method_name == 'combined':
            continue
        detections = [r[1] for r in results[method_name]]
        total_detections = sum(detections)
        detection_rate = total_detections / len(detections) * 100 if detections else 0

        print(f"{method_name:15s}: {total_detections:3d} detections ({detection_rate:5.1f}%)")

    # Combined method summary
    detections = [r[1] for r in results['combined']]
    total_detections = sum(detections)
    detection_rate = total_detections / len(detections) * 100 if detections else 0
    print(f"{'combined':15s}: {total_detections:3d} detections ({detection_rate:5.1f}%)")

    # Vote distribution for combined method
    vote_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, details in results['combined_details']:
        votes = details['votes']
        vote_counts[votes] += 1

    print("\nCombined method vote distribution:")
    for votes, count in vote_counts.items():
        percentage = count / len(results['combined_details']) * 100 if results['combined_details'] else 0
        print(f"  {votes} votes: {count:3d} frames ({percentage:5.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 splash_methods_test.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)

    test_splash_methods(video_path)
