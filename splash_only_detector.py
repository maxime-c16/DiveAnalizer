#!/usr/bin/env python3
"""
Splash-Only Dive Detection System
=================================

A complete shift from state-machine based detection to pure splash detection.
This system uses only splash detection with Gaussian filtering to identify dive endings,
then extracts 10s before + 2s after the splash for complete dive capture.

Key Features:
- Gaussian filtering for false positive reduction
- Adaptive thresholding based on score statistics
- Peak detection for robust splash event identification
- Comprehensive debugging visualizations
- Zone-based splash detection
- Temporal consistency validation

Author: Dive Analysis System
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
import json
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import warnings

# Scientific computing imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    from scipy import ndimage
    from scipy.signal import find_peaks, savgol_filter
    SCIPY_AVAILABLE = True
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some scientific packages not available: {e}")
    SCIPY_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False

# Import existing splash detection methods from slAIcer
try:
    from slAIcer import (
        detect_splash_motion_intensity,
        detect_splash_frame_diff,
        detect_splash_optical_flow,
        detect_splash_contour_analysis,
        detect_splash_combined,
        get_video_fps,
        frame_generator
    )
    print("✅ Successfully imported splash detection methods from slAIcer.py")
except ImportError as e:
    print(f"❌ Failed to import from slAIcer.py: {e}")
    print("Please ensure slAIcer.py is in the same directory.")
    sys.exit(1)

@dataclass
class SplashEvent:
    """Represents a detected splash event with metadata"""
    frame_idx: int
    timestamp: float  # seconds from video start
    score: float
    filtered_score: float
    confidence: str
    zone_info: Dict[str, Any]
    detection_method: str

@dataclass
class DetectionConfig:
    """Configuration for splash detection parameters"""
    # Splash detection method
    method: str = 'motion_intensity'  # 'motion_intensity', 'combined', 'frame_diff'

    # Zone definition (normalized coordinates)
    splash_zone_top_norm: float = 0.7
    splash_zone_bottom_norm: float = 0.95

    # Gaussian filtering parameters
    spatial_gaussian_kernel: Tuple[int, int] = (5, 5)
    temporal_gaussian_sigma: float = 1.5
    temporal_window_size: int = 15  # frames for temporal smoothing

    # Threshold parameters
    base_threshold: float = 12.0
    adaptive_threshold_factor: float = 1.2
    min_threshold: float = 10.0  # Increased from 8.0 to reduce false positives
    max_threshold: float = 25.0
    
    # Event validation parameters
    min_extraction_score: float = 15.0  # Minimum score to extract a dive
    high_confidence_threshold: float = 20.0  # Threshold for high confidence events
    medium_confidence_threshold: float = 12.0  # Threshold for medium confidence events

    # Peak detection parameters
    min_peak_prominence: float = 5.0  # Increased from 3.0 for better filtering
    min_peak_distance: int = 30  # minimum frames between peaks
    peak_width_range: Tuple[int, int] = (3, 20)  # min and max peak width

    # Temporal consistency
    min_sustained_frames: int = 3  # minimum frames above threshold
    cooldown_frames: int = 60  # frames to wait after detection

    # Video extraction parameters
    pre_splash_duration: float = 5.0  # seconds before splash
    post_splash_duration: float = 2.0   # seconds after splash

    # Debugging
    enable_debug_plots: bool = True
    save_debug_frames: bool = False
    debug_output_dir: str = "debug_splash_detection"

class GaussianSplashFilter:
    """
    Advanced Gaussian filtering system for splash score smoothing and false positive reduction
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.score_history = deque(maxlen=config.temporal_window_size * 3)  # Extended history
        self.filtered_history = deque(maxlen=config.temporal_window_size * 3)
        self.timestamp_history = deque(maxlen=config.temporal_window_size * 3)

        # Statistics for adaptive thresholding
        self.recent_scores = deque(maxlen=100)  # For statistical analysis
        self.background_score_estimate = 0.0

    def add_score(self, score: float, timestamp: float) -> float:
        """
        Add a new score and return the Gaussian-filtered version
        """
        self.score_history.append(score)
        self.timestamp_history.append(timestamp)
        self.recent_scores.append(score)

        # Update background estimate (rolling median for robustness)
        if len(self.recent_scores) >= 20:
            sorted_recent = sorted(list(self.recent_scores)[-50:])  # Recent 50 scores
            self.background_score_estimate = sorted_recent[len(sorted_recent) // 2]  # Median

        # Apply temporal Gaussian smoothing
        filtered_score = self._apply_temporal_gaussian()
        self.filtered_history.append(filtered_score)

        return filtered_score

    def _apply_temporal_gaussian(self) -> float:
        """
        Apply Gaussian smoothing over temporal sequence of scores
        """
        if len(self.score_history) < 3:
            return self.score_history[-1] if self.score_history else 0.0

        # Convert to numpy array for processing
        scores = np.array(list(self.score_history))

        # Apply Gaussian filter using scipy if available
        if SCIPY_AVAILABLE:
            # Use gaussian_filter1d for temporal smoothing
            sigma = self.config.temporal_gaussian_sigma
            filtered = ndimage.gaussian_filter1d(scores, sigma=sigma, mode='nearest')
            return float(filtered[-1])  # Return the most recent filtered value
        else:
            # Fallback: simple moving average with Gaussian-like weights
            window_size = min(len(scores), self.config.temporal_window_size)
            if window_size < 3:
                return float(scores[-1])

            # Create Gaussian-like weights
            weights = np.exp(-0.5 * np.linspace(-2, 2, window_size)**2)
            weights = weights / np.sum(weights)

            # Apply weighted average to recent scores
            recent_scores = scores[-window_size:]
            return float(np.sum(recent_scores * weights))

    def get_adaptive_threshold(self) -> float:
        """
        Calculate adaptive threshold based on recent score statistics
        """
        if len(self.recent_scores) < 10:
            return self.config.base_threshold

        # Statistical analysis of recent scores
        recent_array = np.array(list(self.recent_scores)[-50:])  # Recent 50 scores
        mean_score = np.mean(recent_array)
        std_score = np.std(recent_array)

        # Adaptive threshold: background + factor * std
        adaptive_thresh = self.background_score_estimate + self.config.adaptive_threshold_factor * std_score

        # Clamp to reasonable bounds
        adaptive_thresh = max(self.config.min_threshold,
                            min(self.config.max_threshold, adaptive_thresh))

        return adaptive_thresh

    def get_score_statistics(self) -> Dict[str, float]:
        """Get current statistics for debugging"""
        if len(self.recent_scores) < 5:
            return {'background': 0.0, 'mean': 0.0, 'std': 0.0, 'adaptive_threshold': self.config.base_threshold}

        recent_array = np.array(list(self.recent_scores)[-50:])
        return {
            'background': self.background_score_estimate,
            'mean': float(np.mean(recent_array)),
            'std': float(np.std(recent_array)),
            'adaptive_threshold': self.get_adaptive_threshold()
        }

class PeakDetector:
    """
    Robust peak detection for identifying splash events in filtered score sequences
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.recent_peaks = deque(maxlen=20)  # Track recent peak locations

    def detect_peaks(self, filtered_scores: List[float], timestamps: List[float]) -> List[int]:
        """
        Detect peaks in the filtered score sequence using sophisticated peak detection
        """
        if len(filtered_scores) < self.config.min_peak_distance:
            return []

        scores_array = np.array(filtered_scores)

        if SCIPY_AVAILABLE:
            # Use scipy's find_peaks for robust detection
            peaks, properties = find_peaks(
                scores_array,
                prominence=self.config.min_peak_prominence,
                distance=self.config.min_peak_distance,
                width=self.config.peak_width_range
            )
            return peaks.tolist()
        else:
            # Fallback: simple peak detection
            peaks = []
            min_distance = self.config.min_peak_distance

            for i in range(min_distance, len(scores_array) - min_distance):
                is_peak = True
                current_score = scores_array[i]

                # Check if it's higher than neighbors within min_distance
                for j in range(i - min_distance, i + min_distance + 1):
                    if j != i and scores_array[j] >= current_score:
                        is_peak = False
                        break

                # Check prominence (height above surrounding area)
                local_min = min(np.min(scores_array[max(0, i-min_distance):i]),
                               np.min(scores_array[i+1:min(len(scores_array), i+min_distance+1)]))
                prominence = current_score - local_min

                if is_peak and prominence >= self.config.min_peak_prominence:
                    peaks.append(i)

            return peaks

class SplashOnlyDetector:
    """
    Main detector class implementing pure splash-based dive detection
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.gaussian_filter = GaussianSplashFilter(config)
        self.peak_detector = PeakDetector(config)

        # Detection state
        self.detected_events: List[SplashEvent] = []
        self.last_detection_frame = -1

        # Debug data collection
        self.debug_data = {
            'frame_indices': [],
            'timestamps': [],
            'raw_scores': [],
            'filtered_scores': [],
            'thresholds': [],
            'detected_peaks': [],
            'statistics': []
        }

        # Create debug output directory
        if self.config.enable_debug_plots:
            os.makedirs(self.config.debug_output_dir, exist_ok=True)

    def detect_splashes_in_video(self, video_path: str) -> List[SplashEvent]:
        """
        Main detection function: processes entire video and returns detected splash events
        """
        print(f"🌊 Starting splash-only detection on: {video_path}")
        print(f"📊 Method: {self.config.method}")
        print(f"🎯 Zone: {self.config.splash_zone_top_norm:.2f} - {self.config.splash_zone_bottom_norm:.2f}")

        video_fps = get_video_fps(video_path)
        print(f"🎬 Video FPS: {video_fps:.1f}")

        # Process all frames
        prev_gray = None
        frame_count = 0
        start_time = time.time()

        for frame_idx, frame in frame_generator(video_path, target_fps=video_fps):
            frame_count += 1
            timestamp = frame_idx / video_fps

            # Progress reporting
            if frame_count % 300 == 0:  # Every 10 seconds at 30fps
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📈 Progress: Frame {frame_idx}, {timestamp:.1f}s, {fps:.1f} fps processing")

            # Extract splash zone
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            band_top = int(self.config.splash_zone_top_norm * h)
            band_bottom = int(self.config.splash_zone_bottom_norm * h)
            splash_band = gray[band_top:band_bottom, :]

            # Apply spatial Gaussian smoothing to reduce noise
            splash_band_smooth = cv2.GaussianBlur(splash_band, self.config.spatial_gaussian_kernel, 0)

            # Calculate splash score
            raw_score = 0.0
            if prev_gray is not None:
                prev_band = prev_gray[band_top:band_bottom, :]
                prev_band_smooth = cv2.GaussianBlur(prev_band, self.config.spatial_gaussian_kernel, 0)

                if self.config.method == 'motion_intensity':
                    _, raw_score = detect_splash_motion_intensity(splash_band_smooth, prev_band_smooth)
                elif self.config.method == 'frame_diff':
                    _, raw_score = detect_splash_frame_diff(splash_band_smooth, prev_band_smooth)
                elif self.config.method == 'optical_flow':
                    _, raw_score = detect_splash_optical_flow(splash_band_smooth, prev_band_smooth)
                elif self.config.method == 'contour':
                    _, raw_score = detect_splash_contour_analysis(splash_band_smooth, prev_band_smooth)
                elif self.config.method == 'combined':
                    _, raw_score, _ = detect_splash_combined(splash_band_smooth, prev_band_smooth)

            # Apply Gaussian temporal filtering
            filtered_score = self.gaussian_filter.add_score(raw_score, timestamp)

            # Get adaptive threshold
            current_threshold = self.gaussian_filter.get_adaptive_threshold()

            # Store debug data
            self.debug_data['frame_indices'].append(frame_idx)
            self.debug_data['timestamps'].append(timestamp)
            self.debug_data['raw_scores'].append(raw_score)
            self.debug_data['filtered_scores'].append(filtered_score)
            self.debug_data['thresholds'].append(current_threshold)
            self.debug_data['statistics'].append(self.gaussian_filter.get_score_statistics())

            # Check for peak detection periodically (every few frames for efficiency)
            if frame_count % 5 == 0:  # Check every 5 frames
                self._check_for_peaks(frame_idx, timestamp, video_fps)

            prev_gray = gray

        # Final peak detection pass
        self._final_peak_detection(video_fps)

        processing_time = time.time() - start_time
        print(f"✅ Detection complete: {len(self.detected_events)} splash events found in {processing_time:.1f}s")

        # Generate debug plots
        if self.config.enable_debug_plots:
            self._generate_debug_plots()

        return self.detected_events

    def _check_for_peaks(self, current_frame: int, current_timestamp: float, video_fps: float):
        """Check for peaks in recent data"""
        if len(self.debug_data['filtered_scores']) < self.config.temporal_window_size:
            return

        # Get recent window for peak detection
        window_size = min(self.config.temporal_window_size * 2, len(self.debug_data['filtered_scores']))
        recent_scores = self.debug_data['filtered_scores'][-window_size:]
        recent_timestamps = self.debug_data['timestamps'][-window_size:]
        recent_frames = self.debug_data['frame_indices'][-window_size:]

        # Detect peaks in recent window
        peaks = self.peak_detector.detect_peaks(recent_scores, recent_timestamps)

        # Process new peaks (only those not already detected)
        for peak_idx in peaks:
            actual_frame_idx = recent_frames[peak_idx]

            # Check if this peak is new and not in cooldown
            if (actual_frame_idx not in [event.frame_idx for event in self.detected_events] and
                actual_frame_idx > self.last_detection_frame + self.config.cooldown_frames):
                
                # Apply stricter validation for event quality
                filtered_score = recent_scores[peak_idx]
                raw_score = self.debug_data['raw_scores'][-window_size + peak_idx]
                
                # Determine confidence based on stricter thresholds
                if filtered_score >= self.config.high_confidence_threshold:
                    confidence = 'high'
                elif filtered_score >= self.config.medium_confidence_threshold:
                    confidence = 'medium'
                else:
                    confidence = 'low'
                
                # Only keep events above minimum extraction threshold or high confidence
                if (filtered_score >= self.config.min_extraction_score or 
                    confidence == 'high'):
                    
                    # Create splash event
                    splash_event = SplashEvent(
                        frame_idx=actual_frame_idx,
                        timestamp=recent_timestamps[peak_idx],
                        score=raw_score,
                        filtered_score=filtered_score,
                        confidence=confidence,
                        zone_info={
                            'top_norm': self.config.splash_zone_top_norm,
                            'bottom_norm': self.config.splash_zone_bottom_norm,
                            'method': self.config.method
                        },
                        detection_method='gaussian_peak'
                    )
                    
                    self.detected_events.append(splash_event)
                    self.last_detection_frame = actual_frame_idx
                    
                    print(f"🌊 SPLASH DETECTED: Frame {actual_frame_idx}, t={splash_event.timestamp:.1f}s, "
                          f"score={splash_event.filtered_score:.1f}, confidence={splash_event.confidence}")
                else:
                    print(f"⚪ Weak event filtered out: Frame {actual_frame_idx}, "
                          f"t={recent_timestamps[peak_idx]:.1f}s, score={filtered_score:.1f} "
                          f"(below extraction threshold {self.config.min_extraction_score})")
    
    def _final_peak_detection(self, video_fps: float):
        """Final pass for peak detection on complete data"""
        if len(self.debug_data['filtered_scores']) < 10:
            return

        # Detect all peaks in complete data
        all_peaks = self.peak_detector.detect_peaks(
            self.debug_data['filtered_scores'],
            self.debug_data['timestamps']
        )

        self.debug_data['detected_peaks'] = all_peaks

        # Validate and add any missed events
        for peak_idx in all_peaks:
            frame_idx = self.debug_data['frame_indices'][peak_idx]

            # Check if this peak was missed in real-time detection
            if not any(event.frame_idx == frame_idx for event in self.detected_events):

                # Apply cooldown check
                too_close = any(abs(event.frame_idx - frame_idx) < self.config.cooldown_frames
                              for event in self.detected_events)

                if not too_close:
                    filtered_score = self.debug_data['filtered_scores'][peak_idx]
                    raw_score = self.debug_data['raw_scores'][peak_idx]
                    
                    # Determine confidence based on stricter thresholds
                    if filtered_score >= self.config.high_confidence_threshold:
                        confidence = 'high'
                    elif filtered_score >= self.config.medium_confidence_threshold:
                        confidence = 'medium'
                    else:
                        confidence = 'low'
                    
                    # Only keep events above minimum extraction threshold or high confidence
                    if (filtered_score >= self.config.min_extraction_score or 
                        confidence == 'high'):
                        
                        splash_event = SplashEvent(
                            frame_idx=frame_idx,
                            timestamp=self.debug_data['timestamps'][peak_idx],
                            score=raw_score,
                            filtered_score=filtered_score,
                            confidence=confidence,
                            zone_info={
                                'top_norm': self.config.splash_zone_top_norm,
                                'bottom_norm': self.config.splash_zone_bottom_norm,
                                'method': self.config.method
                            },
                            detection_method='final_pass'
                        )
                        
                        self.detected_events.append(splash_event)
                        print(f"🌊 ADDITIONAL SPLASH (final pass): Frame {frame_idx}, "
                              f"t={splash_event.timestamp:.1f}s, score={splash_event.filtered_score:.1f}, "
                              f"confidence={confidence}")
                    else:
                        print(f"⚪ Weak final pass event filtered out: Frame {frame_idx}, "
                              f"t={self.debug_data['timestamps'][peak_idx]:.1f}s, "
                              f"score={filtered_score:.1f} (below threshold)")        # Sort events by frame index
        self.detected_events.sort(key=lambda x: x.frame_idx)

    def _generate_debug_plots(self):
        """Generate comprehensive debug visualization plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available - skipping debug plots")
            return

        print("📊 Generating debug plots...")

        # Create figure with multiple subplots
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'Splash Detection Analysis - Method: {self.config.method}', fontsize=16, fontweight='bold')

        timestamps = np.array(self.debug_data['timestamps'])
        raw_scores = np.array(self.debug_data['raw_scores'])
        filtered_scores = np.array(self.debug_data['filtered_scores'])
        thresholds = np.array(self.debug_data['thresholds'])

        # Plot 1: Raw vs Filtered Scores
        axes[0].plot(timestamps, raw_scores, 'lightblue', alpha=0.7, linewidth=0.8, label='Raw Scores')
        axes[0].plot(timestamps, filtered_scores, 'blue', linewidth=2, label='Gaussian Filtered')
        axes[0].plot(timestamps, thresholds, 'red', linewidth=1.5, linestyle='--', label='Adaptive Threshold')

        # Mark detected events
        for event in self.detected_events:
            axes[0].axvline(x=event.timestamp, color='green', linestyle='-', alpha=0.8, linewidth=2)
            axes[0].annotate(f'Splash\n{event.confidence}',
                           xy=(event.timestamp, event.filtered_score),
                           xytext=(event.timestamp, event.filtered_score + 5),
                           arrowprops=dict(arrowstyle='->', color='green'),
                           fontsize=8, ha='center')

        axes[0].set_title('Score Comparison and Event Detection')
        axes[0].set_ylabel('Splash Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Peak Detection Visualization
        axes[1].plot(timestamps, filtered_scores, 'blue', linewidth=2, label='Filtered Scores')

        # Mark all detected peaks
        if self.debug_data['detected_peaks']:
            peak_timestamps = [timestamps[i] for i in self.debug_data['detected_peaks']]
            peak_scores = [filtered_scores[i] for i in self.debug_data['detected_peaks']]
            axes[1].scatter(peak_timestamps, peak_scores, color='red', s=50, marker='o',
                          label='Detected Peaks', zorder=5)

        # Mark final splash events
        for event in self.detected_events:
            axes[1].scatter(event.timestamp, event.filtered_score, color='green', s=100,
                          marker='*', zorder=6)

        axes[1].set_title('Peak Detection Results')
        axes[1].set_ylabel('Filtered Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Threshold Adaptation
        axes[2].plot(timestamps, thresholds, 'red', linewidth=2, label='Adaptive Threshold')

        # Extract background estimates
        background_estimates = [stat['background'] for stat in self.debug_data['statistics']]
        axes[2].plot(timestamps, background_estimates, 'orange', linewidth=1.5,
                    label='Background Estimate')

        axes[2].set_title('Threshold Adaptation Over Time')
        axes[2].set_ylabel('Threshold Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Statistical Analysis
        means = [stat['mean'] for stat in self.debug_data['statistics']]
        stds = [stat['std'] for stat in self.debug_data['statistics']]

        axes[3].plot(timestamps, means, 'purple', linewidth=1.5, label='Mean Score')
        axes[3].fill_between(timestamps,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.3, color='purple', label='±1 Std Dev')

        axes[3].set_title('Score Statistics Over Time')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_ylabel('Score Statistics')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.debug_output_dir, 'splash_detection_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Debug plot saved: {plot_path}")

        # Save detailed plot data
        data_path = os.path.join(self.config.debug_output_dir, 'debug_data.json')
        debug_export = {
            'config': self.config.__dict__,
            'detection_summary': {
                'total_events': len(self.detected_events),
                'events': [
                    {
                        'frame': event.frame_idx,
                        'timestamp': event.timestamp,
                        'score': event.score,
                        'filtered_score': event.filtered_score,
                        'confidence': event.confidence,
                        'method': event.detection_method
                    }
                    for event in self.detected_events
                ]
            },
            'statistics': {
                'total_frames': len(self.debug_data['frame_indices']),
                'avg_raw_score': float(np.mean(raw_scores)),
                'avg_filtered_score': float(np.mean(filtered_scores)),
                'score_std': float(np.std(raw_scores)),
                'filtered_score_std': float(np.std(filtered_scores))
            }
        }

        with open(data_path, 'w') as f:
            json.dump(debug_export, f, indent=2)
        print(f"📄 Debug data saved: {data_path}")

        plt.show()

def get_splash_zone_interactive(video_path: str) -> Tuple[float, float]:
    """
    Interactive splash zone selection using mouse click and drag.
    Returns (top_norm, bottom_norm) normalized coordinates for the splash zone.
    """
    print("🎯 Opening interactive splash zone selector...")

    # Get first frame from video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Cannot read first frame from video")

    cap.release()

    h, w = frame.shape[:2]
    clicked = {'top_y': None, 'bottom_y': None}
    mouse_pos = {'y': None, 'changed': False}
    drawing = {'active': False, 'start_y': None}

    # Pre-create a resized version for faster display if frame is large
    display_scale = 1.0
    if frame.shape[0] > 1080 or frame.shape[1] > 1920:
        display_scale = min(1080 / frame.shape[0], 1920 / frame.shape[1])

    if display_scale < 1.0:
        display_height = int(frame.shape[0] * display_scale)
        display_width = int(frame.shape[1] * display_scale)
        display_frame = cv2.resize(frame, (display_width, display_height))
    else:
        display_frame = frame.copy()

    clone = display_frame.copy()

    def on_mouse(event, x, y, _flags, _param):
        # Scale coordinates back to original frame
        actual_y = int(y / display_scale) if display_scale < 1.0 else y

        if mouse_pos['y'] != actual_y:
            mouse_pos['y'] = actual_y
            mouse_pos['changed'] = True

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing['active'] = True
            drawing['start_y'] = actual_y
        elif event == cv2.EVENT_LBUTTONUP and drawing['active']:
            drawing['active'] = False
            top_y = min(drawing['start_y'], actual_y)
            bottom_y = max(drawing['start_y'], actual_y)
            clicked['top_y'] = top_y
            clicked['bottom_y'] = bottom_y
            cv2.destroyWindow("Select Splash Zone")

    cv2.namedWindow("Select Splash Zone", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Splash Zone", on_mouse)
    print("🖱️  Click and drag to select the splash detection zone.")
    print("📍 The zone should cover the area where water entry splashes occur.")
    print("💡 Recommended: Select area around the water surface.")
    print("⌨️  Press ESC to use default zone (0.7 - 0.95)")

    last_state = None
    while clicked['top_y'] is None or clicked['bottom_y'] is None:
        # Only redraw if something changed
        current_state = (mouse_pos['y'], drawing['active'], drawing['start_y'])
        if mouse_pos['changed'] or last_state != current_state:
            display = clone.copy()

            # Show current mouse position line
            if mouse_pos['y'] is not None:
                display_y = int(mouse_pos['y'] * display_scale) if display_scale < 1.0 else mouse_pos['y']
                cv2.line(display, (0, display_y), (display.shape[1], display_y), (128, 128, 128), 1)

            # Show selection rectangle while dragging
            if drawing['active'] and drawing['start_y'] is not None and mouse_pos['y'] is not None:
                display_start_y = int(drawing['start_y'] * display_scale) if display_scale < 1.0 else drawing['start_y']
                display_mouse_y = int(mouse_pos['y'] * display_scale) if display_scale < 1.0 else mouse_pos['y']

                top = min(display_start_y, display_mouse_y)
                bottom = max(display_start_y, display_mouse_y)

                # Draw the splash zone rectangle
                overlay = display.copy()
                cv2.rectangle(overlay, (0, top), (display.shape[1], bottom), (255, 0, 255), 2)
                cv2.rectangle(overlay, (0, top), (display.shape[1], bottom), (255, 0, 255), -1)
                cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

                # Calculate actual height in original frame coordinates
                actual_height = abs(drawing['start_y'] - mouse_pos['y'])
                cv2.putText(display, f"Splash Zone (height: {actual_height}px)",
                           (20, top-10 if top > 30 else bottom+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # Show instructions
            cv2.putText(display, "Click and drag to select splash zone (ESC for default)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Recommended: Select area around water surface",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(display, f"Frame size: {w}x{h}",
                       (10, display.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Select Splash Zone", display)

            mouse_pos['changed'] = False
            last_state = current_state

        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC to use default
            print("⌨️  ESC pressed - using default splash zone")
            break

    cv2.destroyAllWindows()

    # Convert to normalized coordinates
    if clicked['top_y'] is not None and clicked['bottom_y'] is not None:
        top_norm = clicked['top_y'] / h
        bottom_norm = clicked['bottom_y'] / h

        # Ensure top < bottom
        if top_norm > bottom_norm:
            top_norm, bottom_norm = bottom_norm, top_norm

        print(f"✅ Splash zone selected: {clicked['top_y']}-{clicked['bottom_y']} pixels")
        print(f"📊 Normalized coordinates: {top_norm:.3f} - {bottom_norm:.3f}")
        print(f"📏 Zone height: {abs(clicked['bottom_y'] - clicked['top_y'])} pixels ({abs(bottom_norm - top_norm)*100:.1f}% of frame)")

        # Show preview of selected zone
        print("🔍 Showing preview of selected zone for 3 seconds...")
        preview_frame = frame.copy()
        zone_top_px = int(top_norm * h)
        zone_bottom_px = int(bottom_norm * h)

        # Draw zone with overlay
        overlay = preview_frame.copy()
        cv2.rectangle(overlay, (0, zone_top_px), (w, zone_bottom_px), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, preview_frame, 0.7, 0, preview_frame)
        cv2.rectangle(preview_frame, (0, zone_top_px), (w, zone_bottom_px), (0, 255, 255), 3)

        # Add labels
        cv2.putText(preview_frame, f"SPLASH DETECTION ZONE", (20, zone_top_px - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(preview_frame, f"Height: {zone_bottom_px - zone_top_px}px ({abs(bottom_norm - top_norm)*100:.1f}%)",
                   (20, zone_bottom_px + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(preview_frame, f"Coordinates: {top_norm:.3f} - {bottom_norm:.3f}",
                   (20, zone_bottom_px + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Resize for display if needed
        if display_scale < 1.0:
            preview_height = int(preview_frame.shape[0] * display_scale)
            preview_width = int(preview_frame.shape[1] * display_scale)
            preview_display = cv2.resize(preview_frame, (preview_width, preview_height))
        else:
            preview_display = preview_frame

        cv2.namedWindow("Zone Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Zone Preview", preview_display)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()

        return top_norm, bottom_norm
    else:
        print("🎯 No splash zone selected - using default (0.7 - 0.95)")
        return 0.7, 0.95

def extract_dive_around_splash(video_path: str, splash_event: SplashEvent,
                              config: DetectionConfig, output_dir: str, dive_number: int) -> str:
    """
    Extract dive video: 10 seconds before splash + 2 seconds after splash
    """
    video_fps = get_video_fps(video_path)

    # Calculate frame ranges
    pre_frames = int(config.pre_splash_duration * video_fps)
    post_frames = int(config.post_splash_duration * video_fps)

    start_frame = max(0, splash_event.frame_idx - pre_frames)
    end_frame = splash_event.frame_idx + post_frames

    # Create output filename
    confidence_suffix = f"_{splash_event.confidence}" if splash_event.confidence != 'high' else ""
    output_filename = f"dive_splash_{dive_number + 1}_t{splash_event.timestamp:.1f}s{confidence_suffix}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    print(f"🎬 Extracting dive {dive_number + 1}: frames {start_frame}-{end_frame} "
          f"({config.pre_splash_duration}s + {config.post_splash_duration}s around splash)")

    # Open video for reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (w, h))

    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    # Extract frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    extracted_frames = 0

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Add overlay showing splash detection zone and event marker
        if current_frame == splash_event.frame_idx:
            # Mark the splash frame
            cv2.putText(frame, f"SPLASH DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Score: {splash_event.filtered_score:.1f}", (50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw splash detection zone
        zone_top = int(config.splash_zone_top_norm * h)
        zone_bottom = int(config.splash_zone_bottom_norm * h)
        cv2.rectangle(frame, (0, zone_top), (w, zone_bottom), (255, 0, 0), 2)
        cv2.putText(frame, "Splash Zone", (10, zone_top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        out.write(frame)
        current_frame += 1
        extracted_frames += 1

    cap.release()
    out.release()

    duration = extracted_frames / video_fps
    print(f"✅ Dive {dive_number + 1} extracted: {output_path} ({extracted_frames} frames, {duration:.1f}s)")

    return output_path

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Splash-Only Dive Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python splash_only_detector.py video.mp4 --method motion_intensity --debug
  python splash_only_detector.py video.mp4 --interactive-zone --debug
  python splash_only_detector.py video.mp4 --zone 0.75 0.9 --threshold 15.0
  python splash_only_detector.py video.mp4 --temporal-sigma 2.0 --peak-prominence 5.0
        """
    )

    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output-dir', '-o', default='extracted_dives_splash',
                       help='Output directory for extracted dives')

    # Detection method
    parser.add_argument('--method', choices=['motion_intensity', 'combined', 'frame_diff', 'optical_flow', 'contour'],
                       default='motion_intensity', help='Splash detection method')

    # Zone configuration
    parser.add_argument('--zone', nargs=2, type=float, metavar=('TOP', 'BOTTOM'),
                       default=[0.7, 0.95], help='Splash zone (normalized coordinates)')
    parser.add_argument('--interactive-zone', action='store_true',
                       help='Use interactive zone selection (mouse click and drag)')

    # Threshold parameters
    parser.add_argument('--threshold', type=float, default=12.0, help='Base detection threshold')
    parser.add_argument('--adaptive-factor', type=float, default=1.2,
                       help='Adaptive threshold factor')
    parser.add_argument('--min-extraction-score', type=float, default=15.0,
                       help='Minimum score required to extract a dive (filters weak events)')
    parser.add_argument('--high-confidence-threshold', type=float, default=20.0,
                       help='Threshold for high confidence events')

    # Gaussian filtering
    parser.add_argument('--temporal-sigma', type=float, default=1.5,
                       help='Temporal Gaussian smoothing sigma')
    parser.add_argument('--temporal-window', type=int, default=15,
                       help='Temporal smoothing window size')

    # Peak detection
    parser.add_argument('--peak-prominence', type=float, default=5.0,
                       help='Minimum peak prominence')
    parser.add_argument('--peak-distance', type=int, default=30,
                       help='Minimum distance between peaks (frames)')

    # Video extraction
    parser.add_argument('--pre-duration', type=float, default=10.0,
                       help='Seconds before splash to include')
    parser.add_argument('--post-duration', type=float, default=2.0,
                       help='Seconds after splash to include')

    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug plots')
    parser.add_argument('--debug-dir', default='debug_splash_detection',
                       help='Debug output directory')
    parser.add_argument('--no-extract', action='store_true',
                       help='Only detect, do not extract videos')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return 1

    # Handle interactive zone selection
    if args.interactive_zone:
        try:
            zone_top, zone_bottom = get_splash_zone_interactive(args.video_path)
        except Exception as e:
            print(f"❌ Error during interactive zone selection: {e}")
            print("🔄 Falling back to default zone (0.7 - 0.95)")
            zone_top, zone_bottom = 0.7, 0.95
    else:
        zone_top, zone_bottom = args.zone[0], args.zone[1]

    # Create configuration
    config = DetectionConfig(
        method=args.method,
        splash_zone_top_norm=zone_top,
        splash_zone_bottom_norm=zone_bottom,
        base_threshold=args.threshold,
        adaptive_threshold_factor=args.adaptive_factor,
        min_extraction_score=args.min_extraction_score,
        high_confidence_threshold=args.high_confidence_threshold,
        medium_confidence_threshold=args.threshold,  # Use base threshold for medium
        temporal_gaussian_sigma=args.temporal_sigma,
        temporal_window_size=args.temporal_window,
        min_peak_prominence=args.peak_prominence,
        min_peak_distance=args.peak_distance,
        pre_splash_duration=args.pre_duration,
        post_splash_duration=args.post_duration,
        enable_debug_plots=args.debug,
        debug_output_dir=args.debug_dir
    )

    print("🌊 Splash-Only Dive Detection System")
    print("====================================")
    print(f"📹 Video: {args.video_path}")
    print(f"🎯 Method: {config.method}")
    print(f"📊 Zone: {config.splash_zone_top_norm:.2f} - {config.splash_zone_bottom_norm:.2f}")
    print(f"🎛️  Threshold: {config.base_threshold} (adaptive factor: {config.adaptive_threshold_factor})")
    print(f"🔍 Gaussian σ: {config.temporal_gaussian_sigma}, window: {config.temporal_window_size}")
    print(f"⛰️  Peak prominence: {config.min_peak_prominence}, distance: {config.min_peak_distance}")
    print(f"✂️  Filtering: min extraction score = {config.min_extraction_score}, high confidence = {config.high_confidence_threshold}")
    print(f"⏱️  Extract: {config.pre_splash_duration}s before + {config.post_splash_duration}s after")

    try:
        # Initialize detector
        detector = SplashOnlyDetector(config)

        # Detect splash events
        start_time = time.time()
        splash_events = detector.detect_splashes_in_video(args.video_path)
        detection_time = time.time() - start_time

        print(f"\n🎯 DETECTION RESULTS")
        print(f"==================")
        print(f"📊 Total splash events detected: {len(splash_events)}")
        print(f"⏱️  Detection time: {detection_time:.1f} seconds")
        
        # Show filtering statistics
        if hasattr(detector, 'debug_data') and detector.debug_data.get('detected_peaks'):
            total_peaks = len(detector.debug_data['detected_peaks'])
            filtered_out = total_peaks - len(splash_events)
            print(f"📈 Peak detection statistics:")
            print(f"   • Total peaks found: {total_peaks}")
            print(f"   • Events extracted: {len(splash_events)}")
            print(f"   • Filtered out (weak): {filtered_out}")
            if total_peaks > 0:
                print(f"   • Extraction rate: {len(splash_events)/total_peaks*100:.1f}%")
        
        if splash_events:
            print("\n🌊 Detected Events:")
            for i, event in enumerate(splash_events):
                print(f"  {i+1}. Frame {event.frame_idx:6d} | "
                      f"t={event.timestamp:6.1f}s | "
                      f"Score={event.filtered_score:6.1f} | "
                      f"Confidence={event.confidence}")
        else:
            print("\n⚠️  No high-quality splash events detected!")
            print(f"💡 Try lowering --min-extraction-score (current: {config.min_extraction_score})")
            print(f"💡 Or check the splash zone selection and detection method")        # Extract dive videos
        if not args.no_extract and splash_events:
            print(f"\n🎬 EXTRACTING DIVE VIDEOS")
            print(f"=========================")

            os.makedirs(args.output_dir, exist_ok=True)

            extraction_start = time.time()
            extracted_paths = []

            for i, event in enumerate(splash_events):
                try:
                    output_path = extract_dive_around_splash(
                        args.video_path, event, config, args.output_dir, i
                    )
                    extracted_paths.append(output_path)
                except Exception as e:
                    print(f"❌ Failed to extract dive {i+1}: {e}")

            extraction_time = time.time() - extraction_start

            print(f"\n✅ EXTRACTION COMPLETE")
            print(f"======================")
            print(f"📄 Extracted videos: {len(extracted_paths)}")
            print(f"⏱️  Extraction time: {extraction_time:.1f} seconds")
            print(f"📁 Output directory: {args.output_dir}")

            if extracted_paths:
                print("\n📹 Created files:")
                for path in extracted_paths:
                    print(f"  • {os.path.basename(path)}")

        return 0

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
