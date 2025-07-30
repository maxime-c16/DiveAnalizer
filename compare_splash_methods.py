#!/usr/bin/env python3
"""
Splash Detection Method Comparison Tool

This script analyzes a video and compares the splash detection scores across all methods:
- frame_diff
- optical_flow
- contour
- motion_intensity
- combined

Generates plots to visualize and compare the effectiveness of each method.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add the current directory to path to import from slAIcer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the splash detection functions from slAIcer
from slAIcer import (
    frame_generator, get_video_fps,
    detect_splash_frame_diff,
    detect_splash_optical_flow,
    detect_splash_contour_analysis,
    detect_splash_motion_intensity,
    detect_splash_combined,
    get_splash_zone,
    suppress_stderr
)

# Suppress MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

def analyze_single_method(method_name, frames_data, splash_top, splash_bottom, threshold=7.0):
    """
    Analyze a single splash detection method on frame data.

    Args:
        method_name: Name of the method to test
        frames_data: List of (frame_idx, gray_frame) tuples
        splash_top, splash_bottom: Splash zone coordinates
        threshold: Detection threshold

    Returns:
        dict: Results for this method
    """
    print(f"  Threading: Starting {method_name} analysis...")

    results = {
        'scores': [],
        'detections': [],
        'details': [] if method_name == 'combined' else None
    }

    # For motion_intensity peak detection
    if method_name == 'motion_intensity':
        splash_event_state = 'waiting'  # 'waiting', 'in_splash', 'post_splash'
        splash_start_idx = None
        peak_score = 0
        frames_since_peak = 0
        cooldown_frames = 5  # Minimum frames between splash events

    prev_band = None

    for i, (frame_idx, gray_frame) in enumerate(frames_data):
        h, w = gray_frame.shape

        # Extract splash detection band
        band_top = max(0, splash_top)
        band_bottom = min(h, splash_bottom)
        splash_band = gray_frame[band_top:band_bottom, :]

        if prev_band is not None:
            # Get the raw score from detection method
            if method_name == 'frame_diff':
                is_splash, score = detect_splash_frame_diff(splash_band, prev_band, threshold)
            elif method_name == 'optical_flow':
                is_splash, score = detect_splash_optical_flow(splash_band, prev_band)
            elif method_name == 'contour':
                is_splash, score = detect_splash_contour_analysis(splash_band, prev_band)
            elif method_name == 'motion_intensity':
                _, score = detect_splash_motion_intensity(splash_band, prev_band)

                # Implement peak-based splash event detection
                is_splash = False

                if splash_event_state == 'waiting':
                    if score > threshold:
                        # Start of new splash event
                        splash_event_state = 'in_splash'
                        splash_start_idx = i
                        peak_score = score
                        print(f"    Motion Intensity: Splash event started at frame {i}, score: {score:.1f}")

                elif splash_event_state == 'in_splash':
                    if score > peak_score:
                        # Update peak
                        peak_score = score
                        frames_since_peak = 0
                    else:
                        frames_since_peak += 1

                    if score <= threshold:
                        # End of splash event - mark the peak frame as detection
                        splash_event_state = 'post_splash'
                        is_splash = True  # This frame marks the end of the splash event
                        print(f"    Motion Intensity: Splash event ended at frame {i}, peak: {peak_score:.1f}")
                        frames_since_peak = 0

                elif splash_event_state == 'post_splash':
                    frames_since_peak += 1
                    if frames_since_peak >= cooldown_frames:
                        splash_event_state = 'waiting'
                        print(f"    Motion Intensity: Ready for next splash detection after cooldown")

            elif method_name == 'combined':
                is_splash, score, details = detect_splash_combined(splash_band, prev_band, threshold)
                results['details'].append(details)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            results['scores'].append(score)
            results['detections'].append(is_splash)

        prev_band = splash_band

    print(f"  Threading: Completed {method_name} analysis ({len(results['scores'])} frames)")
    return method_name, results

def analyze_splash_methods(video_path, max_frames=1000, threshold=7.0):
    """
    Analyze all splash detection methods on a video using threading.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to analyze
        threshold: Threshold for frame_diff and combined methods

    Returns:
        dict: Dictionary containing scores for each method
    """
    print(f"Analyzing splash detection methods on: {video_path}")
    print(f"Processing up to {max_frames} frames...")

    # Get video info
    video_fps = get_video_fps(video_path)
    print(f"Video framerate: {video_fps:.1f}fps")

    # Get first frame and let user select splash zone
    print("Getting first frame for splash zone selection...")
    try:
        _, first_frame = next(frame_generator(video_path))
        print("Please select the splash detection zone:")
        splash_top_px, splash_bottom_px = get_splash_zone(first_frame)
        if splash_top_px is None or splash_bottom_px is None:
            print("No splash zone selected. Using default zone (bottom 20% of frame).")
            h = first_frame.shape[0]
            splash_zone_coords = (int(h * 0.8), h)
        else:
            splash_zone_coords = (splash_top_px, splash_bottom_px)
    except StopIteration:
        raise ValueError("Could not read first frame from video")

    splash_top, splash_bottom = splash_zone_coords
    print(f"Using splash zone: {splash_top} to {splash_bottom} pixels")

    # Load all frames into memory for threading
    print("Loading frames into memory...")
    frames_data = []
    frame_indices = []

    frame_count = 0
    for idx, frame in frame_generator(video_path):
        if frame_count >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_data.append((idx, gray))
        frame_indices.append(idx)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Loaded {frame_count} frames...")

    print(f"Loaded {len(frames_data)} frames. Starting threaded analysis...")

    # Methods to analyze
    methods = ['frame_diff', 'optical_flow', 'contour', 'motion_intensity', 'combined']

    # Run analysis in parallel using threads
    results = {
        'frames': frame_indices[1:],  # Skip first frame (no previous for comparison)
        'frame_diff': {'scores': [], 'detections': []},
        'optical_flow': {'scores': [], 'detections': []},
        'contour': {'scores': [], 'detections': []},
        'motion_intensity': {'scores': [], 'detections': []},
        'combined': {'scores': [], 'detections': [], 'details': []}
    }

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all method analyses
        future_to_method = {
            executor.submit(analyze_single_method, method, frames_data, splash_top, splash_bottom, threshold): method
            for method in methods
        }

        # Collect results as they complete
        for future in as_completed(future_to_method):
            method_name, method_results = future.result()

            results[method_name]['scores'] = method_results['scores']
            results[method_name]['detections'] = method_results['detections']

            if method_name == 'combined':
                results[method_name]['details'] = method_results['details']

    analysis_time = time.time() - start_time
    print(f"Threaded analysis complete in {analysis_time:.1f}s. Processed {len(results['frames'])} frames.")

    return results, video_fps, splash_zone_coords

def plot_splash_comparison(results, video_fps, splash_zone_coords, output_dir=None, threshold=7.0):
    """
    Create comparison plots showing all splash detection methods on the same graph.

    Args:
        results: Results dictionary from analyze_splash_methods
        video_fps: Video framerate for time axis
        splash_zone_coords: Splash zone coordinates for title
        output_dir: Directory to save plots (optional)
        threshold: Detection threshold for visualization
    """
    frames = np.array(results['frames'])
    time_axis = frames / video_fps  # Convert to seconds

    # Create figure with 2 subplots: raw scores and normalized scores
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f'Splash Detection Method Comparison\nSplash Zone: {splash_zone_coords[0]}-{splash_zone_coords[1]}px | Threshold: {threshold}',
                 fontsize=14, fontweight='bold')

    methods = ['frame_diff', 'optical_flow', 'contour', 'motion_intensity', 'combined']
    # Use distinct colors that are easily distinguishable
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']

    # Plot 1: Raw scores with different scales
    print("Creating raw scores plot...")
    for i, (method, color, style, marker) in enumerate(zip(methods, colors, line_styles, markers)):
        scores = np.array(results[method]['scores'])
        detections = np.array(results[method]['detections'])

        # Plot the score line
        ax1.plot(time_axis, scores, color=color, alpha=0.8, linewidth=2,
                linestyle=style, label=f'{method.replace("_", " ").title()}')

        # Mark detections with markers
        detection_indices = np.where(detections)[0]
        if len(detection_indices) > 0:
            ax1.scatter(time_axis[detection_indices], scores[detection_indices],
                       color=color, s=30, alpha=0.9, marker=marker,
                       edgecolors='black', linewidth=0.5, zorder=5)

    # Add threshold line for relevant methods
    ax1.axhline(y=threshold, color='red', linestyle=':', alpha=0.7, linewidth=2,
               label=f'Detection Threshold ({threshold})')

    ax1.set_title('Raw Splash Detection Scores', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Detection Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Plot 2: Normalized scores (0-1) for better comparison
    print("Creating normalized scores plot...")
    for i, (method, color, style, marker) in enumerate(zip(methods, colors, line_styles, markers)):
        scores = np.array(results[method]['scores'])
        detections = np.array(results[method]['detections'])

        # Normalize scores to 0-1 range
        if np.max(scores) > 0:
            normalized_scores = scores / np.max(scores)
        else:
            normalized_scores = scores

        # Plot the normalized score line
        ax2.plot(time_axis, normalized_scores, color=color, alpha=0.8, linewidth=2,
                linestyle=style, label=f'{method.replace("_", " ").title()}')

        # Mark detections with markers
        detection_indices = np.where(detections)[0]
        if len(detection_indices) > 0:
            ax2.scatter(time_axis[detection_indices], normalized_scores[detection_indices],
                       color=color, s=30, alpha=0.9, marker=marker,
                       edgecolors='black', linewidth=0.5, zorder=5)

    ax2.set_title('Normalized Splash Detection Scores (0-1 Scale)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Normalized Score')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.9)

    # Add detection count information to both plots
    for ax in [ax1, ax2]:
        # Add text box with detection counts
        detection_text = "Detections: "
        for method in methods:
            count = np.sum(results[method]['detections'])
            detection_text += f"{method.replace('_', ' ').title()}: {count}, "
        detection_text = detection_text.rstrip(", ")

        ax.text(0.02, 0.98, detection_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=9)

    plt.tight_layout()

    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'splash_methods_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

    plt.show()

def print_detection_summary(results, video_fps):
    """
    Print a summary of detection performance for each method.
    """
    print("\n" + "="*60)
    print("SPLASH DETECTION SUMMARY")
    print("="*60)

    total_frames = len(results['frames'])
    duration = total_frames / video_fps

    print(f"Video Duration: {duration:.1f} seconds ({total_frames} frames)")
    print(f"Video FPS: {video_fps:.1f}")
    print()

    methods = ['frame_diff', 'optical_flow', 'contour', 'motion_intensity', 'combined']

    print(f"{'Method':<15} {'Detections':<12} {'Rate (%)':<10} {'Avg Score':<12} {'Max Score':<12}")
    print("-" * 70)

    for method in methods:
        detections = np.array(results[method]['detections'])
        scores = np.array(results[method]['scores'])

        detection_count = np.sum(detections)
        detection_rate = (detection_count / total_frames) * 100
        avg_score = np.mean(scores)
        max_score = np.max(scores)

        print(f"{method.replace('_', ' ').title():<15} {detection_count:<12} {detection_rate:<10.1f} {avg_score:<12.2f} {max_score:<12.2f}")

    # Combined method voting details
    if 'details' in results['combined']:
        print(f"\nCombined Method Voting Details:")
        details = results['combined']['details']
        total_votes = len([d for d in details if d])
        if total_votes > 0:
            avg_votes = np.mean([d['votes'] for d in details if d])
            print(f"  Average votes per frame: {avg_votes:.2f}")

            # Count voting patterns
            vote_counts = {}
            for detail in details:
                if detail:
                    votes = detail['votes']
                    vote_counts[votes] = vote_counts.get(votes, 0) + 1

            print("  Vote distribution:")
            for votes in sorted(vote_counts.keys()):
                count = vote_counts[votes]
                percentage = (count / total_votes) * 100
                print(f"    {votes} votes: {count} frames ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Compare splash detection methods on a video")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--max_frames", type=int, default=1000,
                       help="Maximum number of frames to analyze (default: 1000)")
    parser.add_argument("--threshold", type=float, default=12.0,
                       help="Detection threshold for frame_diff and combined methods (default: 12.0)")
    parser.add_argument("--output_dir", default="splash_analysis_output",
                       help="Directory to save analysis plots (default: splash_analysis_output)")

    args = parser.parse_args()

    # Validate video path
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    try:
        # Analyze all methods with threading
        results, video_fps, splash_zone_coords = analyze_splash_methods(
            args.video_path,
            max_frames=args.max_frames,
            threshold=args.threshold
        )

        # Print summary
        print_detection_summary(results, video_fps)

        # Create plots
        plot_splash_comparison(results, video_fps, splash_zone_coords,
                             output_dir=args.output_dir, threshold=args.threshold)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
