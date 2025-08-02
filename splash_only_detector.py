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

# Progress bar import (optional)
try:
	from tqdm import tqdm
	TQDM_AVAILABLE = True
except ImportError:
	TQDM_AVAILABLE = False
	# Fallback progress bar class
	class tqdm:
		def __init__(self, iterable=None, total=None, desc=None, unit='it', **kwargs):
			self.iterable = iterable
			self.total = total
			self.desc = desc or ''
			self.unit = unit
			self.n = 0
			self.last_print_n = 0
			self.start_time = time.time()

		def __iter__(self):
			if self.iterable is not None:
				for item in self.iterable:
					yield item
					self.update(1)

		def __enter__(self):
			return self

		def __exit__(self, *args):
			self.close()

		def update(self, n=1):
			self.n += n
			# Print progress every 10% or every 300 frames
			if self.total and (self.n - self.last_print_n >= max(1, self.total // 10) or
							 self.n - self.last_print_n >= 300):
				elapsed = time.time() - self.start_time
				rate = self.n / elapsed if elapsed > 0 else 0
				percent = (self.n / self.total) * 100 if self.total else 0
				print(f"\r{self.desc}: {percent:.1f}% ({self.n}/{self.total}) "
					  f"[{rate:.1f}{self.unit}/s]", end='', flush=True)
				self.last_print_n = self.n

		def close(self):
			if self.total:
				print()  # New line at end

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
	splash_zone_left_norm: float = 0.0
	splash_zone_right_norm: float = 1.0

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
	auto_extract_threshold: bool = False  # Auto-calculate extraction threshold
	high_confidence_threshold: float = 20.0  # Threshold for high confidence events
	medium_confidence_threshold: float = 12.0  # Threshold for medium confidence events
	allow_close_dives: bool = False  # Allow detection of dives closer than 5 seconds

	# Peak detection parameters
	min_peak_prominence: float = 5.0  # Increased from 3.0 for better filtering
	min_peak_distance: int = 30  # minimum frames between peaks
	peak_width_range: Tuple[int, int] = (3, 20)  # min and max peak width

	# Temporal consistency
	min_sustained_frames: int = 3  # minimum frames above threshold
	cooldown_frames: int = 60  # frames to wait after detection

	# Video extraction parameters
	pre_splash_duration: float = 6.0  # seconds before splash
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

class GlobalContextAnalyzer:
	"""
	Intelligent final pass analyzer using complete video statistics
	to make refined decisions about which events are true dives
	"""

	def __init__(self, config: DetectionConfig):
		self.config = config

	def analyze_and_filter_events(self, events: List[SplashEvent], debug_data: Dict) -> List[SplashEvent]:
		"""
		Perform intelligent final pass analysis using global video context
		"""
		if len(events) < 2:
			return events

		print(f"\n🧠 GLOBAL CONTEXT ANALYSIS")
		print(f"===========================")

		# Extract statistics from complete video
		stats_list = debug_data.get('statistics', [])
		all_filtered_scores = debug_data.get('filtered_scores', [])
		all_timestamps = debug_data.get('timestamps', [])

		if not all_filtered_scores:
			print("⚠️  No debug data available for global analysis")
			return events

		# Calculate video-wide statistics from the complete data
		background_score = np.mean(all_filtered_scores)
		noise_level = np.std(all_filtered_scores)

		print(f"📊 Video statistics:")
		print(f"   • Background level: {background_score:.2f}")
		print(f"   • Noise level (std): {noise_level:.2f}")
		print(f"   • Total frames analyzed: {len(all_filtered_scores)}")

		# Calculate dynamic thresholds based on video content
		dynamic_thresholds = self._calculate_dynamic_thresholds(
			events, background_score, noise_level, all_filtered_scores
		)

		# Auto-adjust extraction threshold if it was using default value
		if hasattr(self.config, 'auto_extract_threshold') and self.config.auto_extract_threshold:
			original_threshold = self.config.min_extraction_score
			self.config.min_extraction_score = dynamic_thresholds['auto_extraction']
			print(f"🤖 AUTO-ADJUSTED extraction threshold: {original_threshold:.1f} → {self.config.min_extraction_score:.1f}")

			# Re-filter events with new threshold if needed
			events = self._refilter_events_with_new_threshold(events)

		# Analyze event patterns
		event_analysis = self._analyze_event_patterns(events, all_timestamps)

		# Score events using multiple criteria
		scored_events = self._score_events_comprehensive(
			events, dynamic_thresholds, event_analysis, all_filtered_scores
		)

		# Apply intelligent filtering
		filtered_events = self._apply_intelligent_filter(scored_events)

		return filtered_events

	def _refilter_events_with_new_threshold(self, events: List[SplashEvent]) -> List[SplashEvent]:
		"""Re-filter events with the new auto-calculated threshold"""
		original_count = len(events)
		filtered_events = [
			event for event in events
			if event.filtered_score >= self.config.min_extraction_score or event.confidence == 'high'
		]
		filtered_count = len(filtered_events)

		if filtered_count != original_count:
			print(f"🔄 Re-filtered with auto threshold: {original_count} → {filtered_count} events")

		return filtered_events

	def _calculate_dynamic_thresholds(self, events: List[SplashEvent],
									background: float, noise: float,
									all_scores: List[float]) -> Dict[str, float]:
		"""Calculate dynamic thresholds based on video content"""

		# Calculate percentiles for better understanding of score distribution
		scores_array = np.array(all_scores)
		percentiles = {
			'p50': np.percentile(scores_array, 50),
			'p75': np.percentile(scores_array, 75),
			'p90': np.percentile(scores_array, 90),
			'p95': np.percentile(scores_array, 95),
			'p99': np.percentile(scores_array, 99)
		}

		# Event score statistics
		event_scores = [e.filtered_score for e in events]
		event_mean = np.mean(event_scores) if event_scores else background
		event_std = np.std(event_scores) if len(event_scores) > 1 else noise

		# Dynamic thresholds based on video context
		# High quality: significantly above video's 95th percentile
		high_quality_threshold = max(percentiles['p95'], event_mean + event_std)

		# Medium quality: above 90th percentile and well above background
		medium_quality_threshold = max(percentiles['p90'], background + 3 * noise)

		# Minimum viable: above 75th percentile or strong outlier
		minimum_viable_threshold = max(percentiles['p75'], background + 2 * noise)

		# AUTO-CALCULATED extraction thresholds based on statistical analysis
		# Conservative auto threshold: background + 4 sigma (very safe)
		auto_conservative_threshold = background + 4 * noise

		# Balanced auto threshold: 90th percentile (good balance)
		auto_balanced_threshold = percentiles['p90']

		# Aggressive auto threshold: 75th percentile (more inclusive)
		auto_aggressive_threshold = percentiles['p75']

		# Smart auto selection based on video characteristics
		score_range = percentiles['p99'] - percentiles['p50']
		if score_range > 15:  # High dynamic range video
			auto_extraction_threshold = auto_balanced_threshold
			auto_mode = "balanced"
		elif noise > 3:  # Noisy video
			auto_extraction_threshold = auto_conservative_threshold
			auto_mode = "conservative"
		else:  # Clean video
			auto_extraction_threshold = auto_aggressive_threshold
			auto_mode = "aggressive"

		thresholds = {
			'high_quality': high_quality_threshold,
			'medium_quality': medium_quality_threshold,
			'minimum_viable': minimum_viable_threshold,
			'background_plus_3sigma': background + 3 * noise,
			'video_p95': percentiles['p95'],
			'video_p90': percentiles['p90'],
			'video_p75': percentiles['p75'],
			# Auto-calculated thresholds
			'auto_extraction': auto_extraction_threshold,
			'auto_conservative': auto_conservative_threshold,
			'auto_balanced': auto_balanced_threshold,
			'auto_aggressive': auto_aggressive_threshold,
			'auto_mode': auto_mode,
			# Statistics for reference
			'score_range': score_range,
			'noise_level': noise
		}

		print(f"🎯 Dynamic thresholds calculated:")
		print(f"   • High quality: {thresholds['high_quality']:.1f}")
		print(f"   • Medium quality: {thresholds['medium_quality']:.1f}")
		print(f"   • Minimum viable: {thresholds['minimum_viable']:.1f}")
		print(f"   • Video 95th percentile: {thresholds['video_p95']:.1f}")
		print(f"🤖 AUTO-CALCULATED extraction threshold: {auto_extraction_threshold:.1f} ({auto_mode} mode)")
		print(f"   • Conservative (bg+4σ): {auto_conservative_threshold:.1f}")
		print(f"   • Balanced (90th %ile): {auto_balanced_threshold:.1f}")
		print(f"   • Aggressive (75th %ile): {auto_aggressive_threshold:.1f}")

		return thresholds

	def _analyze_event_patterns(self, events: List[SplashEvent],
							   all_timestamps: List[float]) -> Dict[str, Any]:
		"""Analyze temporal patterns in detected events"""

		if len(events) < 2:
			return {'isolated_events': len(events), 'sequences': []}

		# Sort events by timestamp
		sorted_events = sorted(events, key=lambda x: x.timestamp)

		# Find diving sequences (events close in time)
		sequences = []
		current_sequence = [sorted_events[0]]

		for i in range(1, len(sorted_events)):
			time_gap = sorted_events[i].timestamp - sorted_events[i-1].timestamp

			# If events are within 2 minutes, consider them part of same sequence
			if time_gap <= 120:  # 2 minutes
				current_sequence.append(sorted_events[i])
			else:
				if len(current_sequence) >= 2:
					sequences.append(current_sequence)
				current_sequence = [sorted_events[i]]

		# Don't forget the last sequence
		if len(current_sequence) >= 2:
			sequences.append(current_sequence)

		# Calculate time gaps between all events
		time_gaps = []
		for i in range(1, len(sorted_events)):
			gap = sorted_events[i].timestamp - sorted_events[i-1].timestamp
			time_gaps.append(gap)

		analysis = {
			'total_events': len(events),
			'sequences': sequences,
			'isolated_events': len(events) - sum(len(seq) for seq in sequences),
			'avg_time_gap': np.mean(time_gaps) if time_gaps else 0,
			'min_time_gap': min(time_gaps) if time_gaps else 0,
			'max_time_gap': max(time_gaps) if time_gaps else 0
		}

		print(f"⏰ Temporal pattern analysis:")
		print(f"   • Diving sequences found: {len(sequences)}")
		print(f"   • Events in sequences: {sum(len(seq) for seq in sequences)}")
		print(f"   • Isolated events: {analysis['isolated_events']}")
		if time_gaps:
			print(f"   • Average time between events: {analysis['avg_time_gap']:.1f}s")

		return analysis

	def _score_events_comprehensive(self, events: List[SplashEvent],
								  thresholds: Dict[str, float],
								  patterns: Dict[str, Any],
								  all_scores: List[float]) -> List[Dict[str, Any]]:
		"""Score each event using multiple criteria with multi-scale detection"""

		scored_events = []

		# Calculate multi-scale context for better 1m vs 3m springboard detection
		score_percentiles = {
			'p25': np.percentile(all_scores, 25),
			'p50': np.percentile(all_scores, 50),
			'p75': np.percentile(all_scores, 75),
			'p90': np.percentile(all_scores, 90),
			'p95': np.percentile(all_scores, 95),
			'p99': np.percentile(all_scores, 99)
		}

		# Local prominence analysis for multi-scale detection
		local_contexts = self._analyze_local_prominence(events, all_scores)

		for event in events:
			score_components = {}
			local_context = local_contexts.get(event.frame_idx, {})

			# 1. ENHANCED: Multi-scale raw quality (0-10 points)
			# Balance absolute score with local prominence for better far-field detection
			absolute_score = event.filtered_score
			local_prominence = local_context.get('local_prominence', 1.0)

			# Hybrid scoring: 60% absolute + 40% local prominence
			hybrid_quality = 0.6 * (absolute_score / score_percentiles['p99']) + 0.4 * local_prominence

			if hybrid_quality >= 0.8 or absolute_score >= thresholds['high_quality']:
				score_components['raw_quality'] = 10
			elif hybrid_quality >= 0.6 or absolute_score >= thresholds['medium_quality']:
				score_components['raw_quality'] = 7
			elif hybrid_quality >= 0.4 or absolute_score >= thresholds['minimum_viable']:
				score_components['raw_quality'] = 5
			else:
				score_components['raw_quality'] = 2

			# 2. ENHANCED: Multi-scale relative strength (0-10 points)
			# Include both global and local relative strength
			global_strength_ratio = event.filtered_score / thresholds['background_plus_3sigma']
			local_strength_ratio = local_context.get('local_strength_ratio', global_strength_ratio)

			# Use the better of global or local strength ratio
			effective_strength_ratio = max(global_strength_ratio, local_strength_ratio)

			if effective_strength_ratio >= 3.0 or local_prominence >= 0.9:
				score_components['relative_strength'] = 10
			elif effective_strength_ratio >= 2.0 or local_prominence >= 0.7:
				score_components['relative_strength'] = 7
			elif effective_strength_ratio >= 1.5 or local_prominence >= 0.5:
				score_components['relative_strength'] = 5
			else:
				score_components['relative_strength'] = 2

			# 3. Sequence context bonus (0-5 points)
			in_sequence = any(event in seq for seq in patterns['sequences'])
			if in_sequence:
				# Find which sequence this event belongs to
				for seq in patterns['sequences']:
					if event in seq:
						# Bonus for being in a sequence with other high-quality events
						seq_avg_score = np.mean([e.filtered_score for e in seq])
						if seq_avg_score >= thresholds['high_quality']:
							score_components['sequence_bonus'] = 5
						elif seq_avg_score >= thresholds['medium_quality']:
							score_components['sequence_bonus'] = 3
						else:
							score_components['sequence_bonus'] = 1
						break
			else:
				score_components['sequence_bonus'] = 0

			# 4. ENHANCED: Multi-scale statistical outlier score (0-5 points)
			# Consider both global and local outlier characteristics
			global_outlier_score = (event.filtered_score - np.mean(all_scores)) / np.std(all_scores)
			local_outlier_score = local_context.get('local_outlier_sigma', global_outlier_score)

			# Use the better of global or local outlier score
			effective_outlier_score = max(global_outlier_score, local_outlier_score)

			if effective_outlier_score >= 4.0 or local_prominence >= 0.95:  # Strong local prominence
				score_components['outlier_strength'] = 5
			elif effective_outlier_score >= 2.5 or local_prominence >= 0.8:
				score_components['outlier_strength'] = 3
			elif effective_outlier_score >= 1.5 or local_prominence >= 0.6:
				score_components['outlier_strength'] = 2
			else:
				score_components['outlier_strength'] = 0

			# Calculate total score
			total_score = sum(score_components.values())

			scored_events.append({
				'event': event,
				'total_score': total_score,
				'components': score_components,
				'strength_ratio': effective_strength_ratio,
				'outlier_sigma': effective_outlier_score,
				'local_prominence': local_prominence,
				'hybrid_quality': hybrid_quality
			})

		# Sort by total score
		scored_events.sort(key=lambda x: x['total_score'], reverse=True)

		return scored_events

	def _analyze_local_prominence(self, events: List[SplashEvent], all_scores: List[float]) -> Dict[int, Dict[str, float]]:
		"""Analyze local prominence of each event for multi-scale detection"""

		local_contexts = {}

		# Convert to arrays for easier manipulation
		all_scores_array = np.array(all_scores)

		for event in events:
			frame_idx = event.frame_idx

			# Define local window around event (±30 seconds at 30fps = ±900 frames)
			window_size = 900  # 30 seconds * 30 fps
			start_idx = max(0, frame_idx - window_size)
			end_idx = min(len(all_scores), frame_idx + window_size)

			# Extract local scores
			local_scores = all_scores_array[start_idx:end_idx]

			if len(local_scores) > 10:  # Ensure we have enough data
				# Calculate local statistics
				local_mean = np.mean(local_scores)
				local_std = np.std(local_scores)
				local_max = np.max(local_scores)
				local_p95 = np.percentile(local_scores, 95)

				# Local prominence: how much this event stands out in its local context
				if local_std > 0:
					local_prominence = min(1.0, (event.filtered_score - local_mean) / (3 * local_std))
					local_prominence = max(0.0, local_prominence)  # Clamp to [0,1]
				else:
					local_prominence = 0.5

				# Local strength ratio: compared to local background
				local_background = local_mean + local_std  # Conservative local background
				local_strength_ratio = event.filtered_score / max(local_background, 1.0)

				# Local outlier sigma
				local_outlier_sigma = (event.filtered_score - local_mean) / max(local_std, 1.0)

				# Peak rank in local context
				local_rank = (local_scores < event.filtered_score).sum() / len(local_scores)

				local_contexts[frame_idx] = {
					'local_prominence': local_prominence,
					'local_strength_ratio': local_strength_ratio,
					'local_outlier_sigma': local_outlier_sigma,
					'local_rank': local_rank,
					'local_max': local_max,
					'local_p95': local_p95
				}
			else:
				# Fallback for edge cases
				local_contexts[frame_idx] = {
					'local_prominence': 0.5,
					'local_strength_ratio': 1.0,
					'local_outlier_sigma': 0.0,
					'local_rank': 0.5,
					'local_max': event.filtered_score,
					'local_p95': event.filtered_score
				}

		return local_contexts

	def _apply_intelligent_filter(self, scored_events: List[Dict[str, Any]]) -> List[SplashEvent]:
		"""Apply intelligent filtering based on comprehensive scores with multi-scale detection"""

		if not scored_events:
			return []

		print(f"\n🏆 EVENT RANKING AND FILTERING (Multi-Scale Detection)")
		print(f"=======================================================")

		# Show all events with their enhanced scores
		for i, item in enumerate(scored_events):
			event = item['event']
			score = item['total_score']
			components = item['components']
			local_prom = item.get('local_prominence', 0.0)
			hybrid_qual = item.get('hybrid_quality', 0.0)

			print(f"#{i+1:2d} | t={event.timestamp:6.1f}s | "
				  f"Score={event.filtered_score:5.1f} | "
				  f"Total={score:2d}/30 | "
				  f"Raw:{components['raw_quality']}/10 "
				  f"Rel:{components['relative_strength']}/10 "
				  f"Seq:{components['sequence_bonus']}/5 "
				  f"Out:{components['outlier_strength']}/5 | "
				  f"LocalProm:{local_prom:.2f} HybridQ:{hybrid_qual:.2f}")

		# ENHANCED: Multi-scale filtering strategy
		filtered_events = []

		# Strategy 1: Always keep top-tier events (score >= 25/30)
		top_tier_threshold = 25

		# Strategy 2: Keep good events that are well-spaced (score >= 18/30) - LOWERED for 3m detection
		good_tier_threshold = 18

		# Strategy 3: Keep decent events with high local prominence (score >= 12/30) - NEW for far-field
		decent_tier_threshold = 12

		# Strategy 4: Keep locally prominent events even if globally weak - NEW
		local_prominence_threshold = 0.8

		print(f"\n🎯 Applying enhanced multi-scale filtering:")

		# First pass: collect top-tier events
		for item in scored_events:
			if item['total_score'] >= top_tier_threshold:
				filtered_events.append(item['event'])
				print(f"✅ TOP TIER: t={item['event'].timestamp:.1f}s (score={item['total_score']}/30)")

		# Second pass: add good events that don't conflict
		for item in scored_events:
			if (item['total_score'] >= good_tier_threshold and
				item['total_score'] < top_tier_threshold):

				# Check if it conflicts with existing events (too close in time)
				conflicts = any(abs(existing.timestamp - item['event'].timestamp) < 8  # Reduced from 10s
							  for existing in filtered_events)

				if not conflicts:
					filtered_events.append(item['event'])
					print(f"✅ GOOD: t={item['event'].timestamp:.1f}s (score={item['total_score']}/30)")
				else:
					print(f"⚠️  SKIPPED: t={item['event'].timestamp:.1f}s (conflicts with better event)")

		# Third pass: add decent events with high local prominence (NEW for 3m springboard)
		for item in scored_events:
			local_prom = item.get('local_prominence', 0.0)
			if ((item['total_score'] >= decent_tier_threshold or local_prom >= local_prominence_threshold) and
				item['total_score'] < good_tier_threshold and
				item['event'] not in filtered_events):

				conflicts = any(abs(existing.timestamp - item['event'].timestamp) < 12  # Larger window for weaker events
							  for existing in filtered_events)

				if not conflicts:
					filtered_events.append(item['event'])
					reason = "high local prominence" if local_prom >= local_prominence_threshold else "decent score"
					print(f"✅ LOCAL PROMINENCE: t={item['event'].timestamp:.1f}s (score={item['total_score']}/30, {reason})")

		# Fourth pass: if we still have very few events, be more lenient (for challenging videos)
		if len(filtered_events) < 2:
			print(f"🔍 Low event count ({len(filtered_events)}), applying lenient mode...")
			for item in scored_events:
				if (item['total_score'] >= 8 and  # Very low threshold
					item['event'] not in filtered_events):

					conflicts = any(abs(existing.timestamp - item['event'].timestamp) < 15
								  for existing in filtered_events)

					if not conflicts:
						filtered_events.append(item['event'])
						print(f"✅ LENIENT: t={item['event'].timestamp:.1f}s (score={item['total_score']}/30) - Low event count mode")

		# Sort by timestamp
		filtered_events.sort(key=lambda x: x.timestamp)

		print(f"\n📊 MULTI-SCALE FILTERING RESULTS:")
		print(f"   • Original events: {len(scored_events)}")
		print(f"   • After enhanced filtering: {len(filtered_events)}")
		print(f"   • Filtered out: {len(scored_events) - len(filtered_events)}")
		if len(filtered_events) > 0:
			score_range = [item['total_score'] for item in scored_events if item['event'] in filtered_events]
			print(f"   • Selected score range: {min(score_range)}-{max(score_range)}/30")

		return filtered_events

class SplashOnlyDetector:
	"""
	Main detector class implementing pure splash-based dive detection
	"""

	def __init__(self, config: DetectionConfig):
		self.config = config
		self.gaussian_filter = GaussianSplashFilter(config)
		self.peak_detector = PeakDetector(config)
		self.global_analyzer = GlobalContextAnalyzer(config)  # Add global analyzer

		# Detection state
		self.detected_events: List[SplashEvent] = []
		self.last_detection_frame = -1

		# Message buffering for clean progress bar
		self.message_buffer: List[str] = []
		self.quiet_mode = False

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

	def _print_or_buffer(self, message: str):
		"""Print message immediately or buffer it if in quiet mode"""
		if self.quiet_mode:
			self.message_buffer.append(message)
		else:
			print(message)

	def _flush_message_buffer(self):
		"""Print all buffered messages and clear the buffer"""
		if self.message_buffer:
			print()  # Add a newline to separate from progress bar
			for message in self.message_buffer:
				print(message)
			self.message_buffer.clear()

	def detect_splashes_in_video_multipass(self, video_path: str) -> List[SplashEvent]:
		"""
		Multi-pass hierarchical detection: Extract dominant splashes first,
		then re-analyze remaining data to detect farther/smaller dives
		"""
		print(f"🔄 MULTI-PASS HIERARCHICAL DETECTION")
		print(f"====================================")

		all_detected_events = []
		pass_number = 1
		max_passes = 3

		# First pass: Standard detection to get all candidate events
		print(f"\n🌊 PASS {pass_number}: Initial detection")
		initial_events = self.detect_splashes_in_video(video_path)

		if not initial_events:
			print("❌ No events detected in initial pass")
			return []

		# Separate events by confidence/score for iterative removal
		high_confidence_events = [e for e in initial_events if e.confidence == 'high' and e.filtered_score >= 25]
		medium_confidence_events = [e for e in initial_events if e.confidence in ['high', 'medium'] and e.filtered_score >= 18]
		all_candidate_events = initial_events.copy()

		print(f"📊 Initial detection: {len(initial_events)} total events")
		print(f"   • High confidence (score ≥25): {len(high_confidence_events)}")
		print(f"   • Medium+ confidence (score ≥18): {len(medium_confidence_events)}")

		# Add high confidence events to final results
		all_detected_events.extend(high_confidence_events)

		# Multi-pass detection: Remove dominant splashes and re-detect
		current_data = self.debug_data.copy()
		removed_events = high_confidence_events.copy()

		for pass_num in range(2, max_passes + 1):
			print(f"\n🔄 PASS {pass_num}: Re-analyzing after removing {len(removed_events)} dominant splashes")

			# Create suppressed score data by masking areas around removed events
			suppressed_scores = self._suppress_dominant_events(current_data, removed_events)

			# Re-analyze with suppressed data to find missed smaller events
			missed_events = self._detect_missed_events(suppressed_scores, current_data, all_detected_events)

			if not missed_events:
				print(f"   ✅ No additional events found in pass {pass_num}")
				break

			print(f"   🎯 Found {len(missed_events)} additional events:")
			for event in missed_events:
				print(f"      • t={event.timestamp:.1f}s, score={event.filtered_score:.1f}, confidence={event.confidence}")

			# Add new events and prepare for next pass
			all_detected_events.extend(missed_events)
			removed_events.extend([e for e in missed_events if e.filtered_score >= 15])  # Only remove medium+ events

		# Sort final events by timestamp
		all_detected_events.sort(key=lambda x: x.timestamp)

		print(f"\n🏆 MULTI-PASS DETECTION COMPLETE")
		print(f"=================================")
		print(f"📊 Total events across all passes: {len(all_detected_events)}")
		print(f"📈 Improvement: +{len(all_detected_events) - len(high_confidence_events)} additional events detected")

		# Update detector's events list for debug plotting
		self.detected_events = all_detected_events

		return all_detected_events

	def _suppress_dominant_events(self, debug_data: Dict, events_to_suppress: List[SplashEvent]) -> List[float]:
		"""
		Create suppressed score data by reducing scores around dominant splash events
		"""
		suppressed_scores = np.array(debug_data['filtered_scores']).copy()
		timestamps = np.array(debug_data['timestamps'])

		# For each dominant event, reduce scores in surrounding temporal window
		suppression_window = 15.0  # seconds to suppress around each dominant event
		suppression_factor = 0.3   # Reduce scores to 30% of original

		for event in events_to_suppress:
			# Find temporal window around this event
			start_time = event.timestamp - suppression_window / 2
			end_time = event.timestamp + suppression_window / 2

			# Find indices in this window
			window_mask = (timestamps >= start_time) & (timestamps <= end_time)

			# Suppress scores in this window (but don't zero them completely)
			suppressed_scores[window_mask] *= suppression_factor

		return suppressed_scores.tolist()

	def _detect_missed_events(self, suppressed_scores: List[float], original_data: Dict,
							 existing_events: List[SplashEvent]) -> List[SplashEvent]:
		"""
		Detect missed events in suppressed score data using lower thresholds
		"""
		# Calculate new statistics on suppressed data
		scores_array = np.array(suppressed_scores)
		background_level = np.mean(scores_array)
		noise_level = np.std(scores_array)

		# Use more aggressive thresholds for missed events
		percentile_75 = np.percentile(scores_array, 75)
		percentile_85 = np.percentile(scores_array, 85)
		adaptive_threshold = max(background_level + 2 * noise_level, percentile_75)

		print(f"   📊 Suppressed data statistics:")
		print(f"      • Background: {background_level:.2f}")
		print(f"      • Noise (std): {noise_level:.2f}")
		print(f"      • 75th percentile: {percentile_75:.2f}")
		print(f"      • Adaptive threshold: {adaptive_threshold:.2f}")

		# Find peaks in suppressed data with adaptive distance
		adaptive_distance = int(2.0 * 30)  # Reduced to 2 seconds for close sequential dives
		if SCIPY_AVAILABLE:
			peaks, properties = find_peaks(
				suppressed_scores,
				height=adaptive_threshold,
				distance=adaptive_distance,  # 2 second minimum distance at 30fps
				prominence=max(1.0, noise_level * 0.5)  # Reduced prominence for smaller events
			)
		else:
			# Fallback: simple peak detection with reduced distance
			peaks = []
			min_distance_frames = adaptive_distance
			for i in range(min_distance_frames, len(suppressed_scores) - min_distance_frames):
				if (suppressed_scores[i] > suppressed_scores[i-1] and
					suppressed_scores[i] > suppressed_scores[i+1] and
					suppressed_scores[i] >= adaptive_threshold):
					# Check minimum distance to other peaks
					too_close = False
					for existing_peak in peaks:
						if abs(i - existing_peak) < min_distance_frames:
							too_close = True
							break
					if not too_close:
						peaks.append(i)

		missed_events = []
		existing_timestamps = {e.timestamp for e in existing_events}

		for peak_idx in peaks:
			timestamp = original_data['timestamps'][peak_idx]
			frame_idx = original_data['frame_indices'][peak_idx]
			filtered_score = suppressed_scores[peak_idx]
			raw_score = original_data['raw_scores'][peak_idx]

			# Use original (unsuppressed) score for confidence determination
			original_filtered_score = original_data['filtered_scores'][peak_idx]

			# Determine confidence
			if original_filtered_score >= 20:
				confidence = 'high'
			elif original_filtered_score >= 12:
				confidence = 'medium'
			else:
				confidence = 'low'

			# Intelligent proximity filtering based on dive quality and configuration
			should_skip = False
			for existing_ts in existing_timestamps:
				time_diff = abs(timestamp - existing_ts)

				# Apply dynamic proximity rules based on configuration and dive confidence
				if not self.config.allow_close_dives:
					# Default behavior: stricter filtering
					if confidence == 'high' and time_diff < 3.0:
						should_skip = True
						break
					elif confidence == 'medium' and time_diff < 4.0:
						should_skip = True
						break
					elif confidence == 'low' and time_diff < 5.0:
						should_skip = True
						break
				else:
					# Close dive detection enabled: more permissive
					if confidence == 'high' and time_diff < 1.5:
						# High confidence dives: allow if >1.5 seconds apart
						should_skip = True
						break
					elif confidence == 'medium' and time_diff < 2.0:
						# Medium confidence dives: allow if >2 seconds apart
						should_skip = True
						break
					elif confidence == 'low' and time_diff < 3.0:
						# Low confidence dives: allow if >3 seconds apart
						should_skip = True
						break

			if should_skip:
				continue
				confidence = 'high'
			elif original_filtered_score >= 12:
				confidence = 'medium'
			else:
				confidence = 'low'

			# Only keep events with reasonable scores
			if original_filtered_score >= 6.0:  # Lower threshold for missed events
				splash_event = SplashEvent(
					frame_idx=frame_idx,
					timestamp=timestamp,
					score=raw_score,
					filtered_score=original_filtered_score,  # Use original score
					confidence=confidence,
					zone_info={
						'top_norm': self.config.splash_zone_top_norm,
						'bottom_norm': self.config.splash_zone_bottom_norm,
						'left_norm': self.config.splash_zone_left_norm,
						'right_norm': self.config.splash_zone_right_norm,
						'method': self.config.method
					},
					detection_method='multi_pass'
				)

				missed_events.append(splash_event)

		return missed_events

	def detect_splashes_in_video(self, video_path: str) -> List[SplashEvent]:
		"""
		Main detection function: processes entire video and returns detected splash events
		"""
		print(f"🌊 Starting splash-only detection on: {video_path}")
		print(f"📊 Method: {self.config.method}")
		print(f"🎯 Zone: top={self.config.splash_zone_top_norm:.2f}, bottom={self.config.splash_zone_bottom_norm:.2f}")
		print(f"📦 Zone: left={self.config.splash_zone_left_norm:.2f}, right={self.config.splash_zone_right_norm:.2f}")

		video_fps = get_video_fps(video_path)
		print(f"🎬 Video FPS: {video_fps:.1f}")

		# Get total frame count for progress bar
		cap = cv2.VideoCapture(video_path)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		cap.release()
		print(f"📹 Total frames: {total_frames:,}")

		# Process all frames with progress bar
		prev_gray = None
		frame_count = 0
		start_time = time.time()

		# Enable quiet mode to buffer messages during progress bar operation
		self.quiet_mode = True

		# Create progress bar
		with tqdm(total=total_frames, desc="🌊 Detecting splashes",
				 unit="frames", ncols=100,
				 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

			for frame_idx, frame in frame_generator(video_path, target_fps=video_fps):
				frame_count += 1
				timestamp = frame_idx / video_fps

				# Extract splash zone (bounding box)
				h, w = frame.shape[:2]
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				# Calculate bounding box coordinates
				band_top = int(self.config.splash_zone_top_norm * h)
				band_bottom = int(self.config.splash_zone_bottom_norm * h)
				band_left = int(self.config.splash_zone_left_norm * w)
				band_right = int(self.config.splash_zone_right_norm * w)

				# Extract the splash detection region
				splash_band = gray[band_top:band_bottom, band_left:band_right]

				# Apply spatial Gaussian smoothing to reduce noise
				splash_band_smooth = cv2.GaussianBlur(splash_band, self.config.spatial_gaussian_kernel, 0)

				# Calculate splash score
				raw_score = 0.0
				if prev_gray is not None:
					prev_band = prev_gray[band_top:band_bottom, band_left:band_right]
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

				# Update progress bar
				pbar.update(1)

				# Update progress bar description with current statistics
				if frame_count % 150 == 0:  # Update every 5 seconds at 30fps
					pbar.set_description(f"🌊 Detecting splashes (Found: {len(self.detected_events)})")

		# Disable quiet mode and flush any buffered messages
		self.quiet_mode = False
		self._flush_message_buffer()

		# Final peak detection pass
		self._final_peak_detection(video_fps)

		# Apply global context analysis for intelligent filtering
		self.detected_events = self.global_analyzer.analyze_and_filter_events(
			self.detected_events, self.debug_data
		)

		processing_time = time.time() - start_time
		print(f"✅ Detection complete: {len(self.detected_events)} splash events found in {processing_time:.1f}s")

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

			# Apply stricter validation for event quality first
			filtered_score = recent_scores[peak_idx]
			raw_score = self.debug_data['raw_scores'][-window_size + peak_idx]

			# Determine confidence based on stricter thresholds
			if filtered_score >= self.config.high_confidence_threshold:
				confidence = 'high'
				adaptive_cooldown = 30  # 1 second for high confidence
			elif filtered_score >= self.config.medium_confidence_threshold:
				confidence = 'medium'
				adaptive_cooldown = 45  # 1.5 seconds for medium confidence
			else:
				confidence = 'low'
				adaptive_cooldown = self.config.cooldown_frames  # Full cooldown for low confidence

			# Check if this peak is new and passes adaptive cooldown
			if (actual_frame_idx not in [event.frame_idx for event in self.detected_events] and
				actual_frame_idx > self.last_detection_frame + adaptive_cooldown):

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
							'left_norm': self.config.splash_zone_left_norm,
							'right_norm': self.config.splash_zone_right_norm,
							'method': self.config.method
						},
						detection_method='gaussian_peak'
					)

					self.detected_events.append(splash_event)
					self.last_detection_frame = actual_frame_idx

					self._print_or_buffer(f"🌊 SPLASH DETECTED: Frame {actual_frame_idx}, t={splash_event.timestamp:.1f}s, "
						  f"score={splash_event.filtered_score:.1f}, confidence={splash_event.confidence}")
				else:
					self._print_or_buffer(f"⚪ Weak event filtered out: Frame {actual_frame_idx}, "
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
								'left_norm': self.config.splash_zone_left_norm,
								'right_norm': self.config.splash_zone_right_norm,
								'method': self.config.method
							},
							detection_method='final_pass'
						)

						self.detected_events.append(splash_event)
						self._print_or_buffer(f"🌊 ADDITIONAL SPLASH (final pass): Frame {frame_idx}, "
							  f"t={splash_event.timestamp:.1f}s, score={splash_event.filtered_score:.1f}, "
							  f"confidence={confidence}")
					else:
						self._print_or_buffer(f"⚪ Weak final pass event filtered out: Frame {frame_idx}, "
							  f"t={self.debug_data['timestamps'][peak_idx]:.1f}s, "
							  f"score={filtered_score:.1f} (below threshold)")        # Sort events by frame index
		self.detected_events.sort(key=lambda x: x.frame_idx)

	def _generate_debug_plots(self):
		"""Generate comprehensive debug visualization plots optimized for debugging"""
		if not MATPLOTLIB_AVAILABLE:
			print("⚠️  Matplotlib not available - skipping debug plots")
			return

		print("📊 Generating enhanced debug plots...")

		# Create figure with optimized layout for debugging
		fig = plt.figure(figsize=(20, 16))
		gs = fig.add_gridspec(4, 2, height_ratios=[2, 1.5, 1.5, 1], width_ratios=[3, 1])

		# Main detection plot (spans top row)
		ax_main = fig.add_subplot(gs[0, :])

		# Secondary plots
		ax_peaks = fig.add_subplot(gs[1, 0])
		ax_stats = fig.add_subplot(gs[1, 1])
		ax_proximity = fig.add_subplot(gs[2, 0])
		ax_confidence = fig.add_subplot(gs[2, 1])
		ax_timeline = fig.add_subplot(gs[3, :])

		fig.suptitle(f'WAVE Enhanced Splash Detection Analysis - Method: {self.config.method}\n'
					f'Events Found: {len(self.detected_events)}',
					fontsize=14, fontweight='bold')

		timestamps = np.array(self.debug_data['timestamps'])
		raw_scores = np.array(self.debug_data['raw_scores'])
		filtered_scores = np.array(self.debug_data['filtered_scores'])
		thresholds = np.array(self.debug_data['thresholds'])

		# MAIN PLOT: Detailed Detection Analysis
		# Background and noise bands
		background_estimates = [stat['background'] for stat in self.debug_data['statistics']]
		noise_levels = [stat['std'] for stat in self.debug_data['statistics']]

		# Fill background + noise bands
		ax_main.fill_between(timestamps,
							np.array(background_estimates) - np.array(noise_levels),
							np.array(background_estimates) + np.array(noise_levels),
							alpha=0.2, color='gray', label='Noise Band (±1σ)')

		ax_main.fill_between(timestamps,
							np.array(background_estimates) + np.array(noise_levels),
							np.array(background_estimates) + 2*np.array(noise_levels),
							alpha=0.15, color='orange', label='Detection Zone (1-2σ)')

		# Plot scores with better styling
		ax_main.plot(timestamps, raw_scores, '#E8E8E8', alpha=0.6, linewidth=0.8, label='Raw Scores')
		ax_main.plot(timestamps, filtered_scores, '#2E86AB', linewidth=2.5, label='Filtered Signal')
		ax_main.plot(timestamps, thresholds, '#F24236', linewidth=2, linestyle='--',
					label='Adaptive Threshold', alpha=0.8)

		# Enhanced event marking with detailed info
		confidence_colors = {'high': '#28A745', 'medium': '#FFC107', 'low': '#6C757D'}
		confidence_markers = {'high': '*', 'medium': 'o', 'low': '^'}

		for i, event in enumerate(self.detected_events):
			color = confidence_colors.get(event.confidence, '#6C757D')
			marker = confidence_markers.get(event.confidence, 'o')

			# Vertical line for event
			ax_main.axvline(x=event.timestamp, color=color, linestyle='-', alpha=0.7, linewidth=2)

			# Event marker with detailed annotation
			ax_main.scatter(event.timestamp, event.filtered_score,
						   color=color, s=120, marker=marker, zorder=10,
						   edgecolors='white', linewidth=2)

			# Detailed annotation with frame and score info
			ax_main.annotate(f'#{i+1}\n{event.timestamp:.1f}s\nF:{event.frame_idx}\nS:{event.filtered_score:.1f}',
						   xy=(event.timestamp, event.filtered_score),
						   xytext=(0, 40), textcoords='offset points',
						   ha='center', va='bottom', fontsize=8,
						   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
						   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

		ax_main.set_title('TARGET Main Detection Analysis with Precise Event Timing', fontsize=12, fontweight='bold')
		ax_main.set_xlabel('Time (seconds) - Hover coordinates for precise timing')
		ax_main.set_ylabel('Splash Detection Score')
		ax_main.legend(loc='upper right')
		ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

		# Enhanced grid for precise timing
		ax_main.set_xticks(np.arange(0, max(timestamps), 30))  # Major ticks every 30s
		ax_main.set_xticks(np.arange(0, max(timestamps), 5), minor=True)  # Minor ticks every 5s
		ax_main.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.3)

		# PEAKS ANALYSIS
		ax_peaks.plot(timestamps, filtered_scores, '#2E86AB', linewidth=1.5, alpha=0.7)

		if self.debug_data['detected_peaks']:
			peak_timestamps = [timestamps[i] for i in self.debug_data['detected_peaks']]
			peak_scores = [filtered_scores[i] for i in self.debug_data['detected_peaks']]
			ax_peaks.scatter(peak_timestamps, peak_scores, color='red', s=40, marker='o',
						   label=f'Peaks ({len(peak_timestamps)})', zorder=5, alpha=0.8)

		# Mark final events vs rejected peaks
		event_times = [e.timestamp for e in self.detected_events]
		for peak_time, peak_score in zip(peak_timestamps, peak_scores):
			if peak_time not in event_times:
				ax_peaks.scatter(peak_time, peak_score, color='orange', s=30, marker='x',
							   alpha=0.6, label='Rejected Peaks' if peak_time == peak_timestamps[0] else "")

		ax_peaks.set_title('SEARCH Peak Detection Analysis')
		ax_peaks.set_ylabel('Score')
		ax_peaks.legend()
		ax_peaks.grid(True, alpha=0.3)

		# STATISTICS SUMMARY
		total_events = len(self.detected_events)
		high_conf = len([e for e in self.detected_events if e.confidence == 'high'])
		medium_conf = len([e for e in self.detected_events if e.confidence == 'medium'])
		low_conf = len([e for e in self.detected_events if e.confidence == 'low'])

		stats_text = f"""CHART Detection Statistics:

Total Events: {total_events}
High Confidence: {high_conf}
Medium Confidence: {medium_conf}
Low Confidence: {low_conf}

Detection Rate: {total_events/(max(timestamps)/60):.1f}/min
Avg Score: {np.mean([e.filtered_score for e in self.detected_events]):.1f}

Background: {np.mean(background_estimates):.1f}
Noise Level: {np.mean(noise_levels):.1f}
Signal/Noise: {np.mean([e.filtered_score for e in self.detected_events])/np.mean(noise_levels):.1f}x"""

		ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
					 fontsize=9, verticalalignment='top', fontfamily='monospace',
					 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
		ax_stats.set_xlim(0, 1)
		ax_stats.set_ylim(0, 1)
		ax_stats.axis('off')

		# PROXIMITY ANALYSIS (for close dive detection)
		if len(self.detected_events) > 1:
			time_diffs = []
			for i in range(1, len(self.detected_events)):
				diff = self.detected_events[i].timestamp - self.detected_events[i-1].timestamp
				time_diffs.append(diff)

			ax_proximity.hist(time_diffs, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
			ax_proximity.axvline(x=5.0, color='red', linestyle='--',
								label='Original 5s limit', alpha=0.8)
			if self.config.allow_close_dives:
				ax_proximity.axvline(x=2.0, color='green', linestyle='--',
									label='New close-dive limit', alpha=0.8)

			ax_proximity.set_title('STOPWATCH Inter-Event Time Analysis')
			ax_proximity.set_xlabel('Time Between Events (seconds)')
			ax_proximity.set_ylabel('Count')
			ax_proximity.legend()
			ax_proximity.grid(True, alpha=0.3)
		else:
			ax_proximity.text(0.5, 0.5, 'Need >=2 events\nfor proximity analysis',
							ha='center', va='center', transform=ax_proximity.transAxes)
			ax_proximity.set_title('STOPWATCH Inter-Event Time Analysis')

		# CONFIDENCE DISTRIBUTION
		conf_counts = {'high': high_conf, 'medium': medium_conf, 'low': low_conf}
		colors = ['#28A745', '#FFC107', '#6C757D']

		if total_events > 0:
			wedges, texts, autotexts = ax_confidence.pie(conf_counts.values(),
														labels=conf_counts.keys(),
														colors=colors, autopct='%1.0f%%',
														startangle=90)
			ax_confidence.set_title('TARGET Confidence Distribution')
		else:
			ax_confidence.text(0.5, 0.5, 'No events detected',
							 ha='center', va='center', transform=ax_confidence.transAxes)

		# TIMELINE VIEW with enhanced event details
		event_y_positions = []
		for i, event in enumerate(self.detected_events):
			y_pos = 0.8 - (i % 3) * 0.3  # Stagger events in 3 rows
			event_y_positions.append(y_pos)

			color = confidence_colors.get(event.confidence, '#6C757D')
			ax_timeline.scatter(event.timestamp, y_pos, s=150, c=color, marker='o',
							   zorder=5, edgecolors='black', linewidth=1)

			# Event number and time
			ax_timeline.annotate(f'{i+1}', xy=(event.timestamp, y_pos),
							   ha='center', va='center', fontweight='bold',
							   fontsize=8, color='white')

			# Detailed info below
			ax_timeline.annotate(f'{event.timestamp:.1f}s\n{event.filtered_score:.1f}',
							   xy=(event.timestamp, y_pos-0.15),
							   ha='center', va='top', fontsize=7)

		ax_timeline.set_xlim(0, max(timestamps))
		ax_timeline.set_ylim(-0.2, 1.2)
		ax_timeline.set_xlabel('Video Timeline (seconds)')
		ax_timeline.set_title('CALENDAR Event Timeline with Detailed Timing')
		ax_timeline.grid(True, alpha=0.3, axis='x')
		ax_timeline.set_yticks([])

		# Add time ruler
		for t in range(0, int(max(timestamps)), 60):
			ax_timeline.axvline(x=t, color='red', alpha=0.3, linestyle=':')
			ax_timeline.text(t, 1.1, f'{t//60}m', ha='center', va='bottom', fontsize=8)

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

def get_splash_zone_interactive(video_path: str) -> Tuple[float, float, float, float]:
	"""
	Interactive splash zone selection using mouse click and drag with bounding box.
	Returns (top_norm, bottom_norm, left_norm, right_norm) normalized coordinates for the splash zone.
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
	selection_complete = False

	while not selection_complete:
		clicked = {'start_x': None, 'start_y': None, 'end_x': None, 'end_y': None}
		mouse_pos = {'x': None, 'y': None, 'changed': False}
		drawing = {'active': False}

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
			actual_x = int(x / display_scale) if display_scale < 1.0 else x
			actual_y = int(y / display_scale) if display_scale < 1.0 else y

			if mouse_pos['x'] != actual_x or mouse_pos['y'] != actual_y:
				mouse_pos['x'] = actual_x
				mouse_pos['y'] = actual_y
				mouse_pos['changed'] = True

			if event == cv2.EVENT_LBUTTONDOWN:
				drawing['active'] = True
				clicked['start_x'] = actual_x
				clicked['start_y'] = actual_y
				clicked['end_x'] = actual_x
				clicked['end_y'] = actual_y
			elif event == cv2.EVENT_MOUSEMOVE and drawing['active']:
				clicked['end_x'] = actual_x
				clicked['end_y'] = actual_y
			elif event == cv2.EVENT_LBUTTONUP and drawing['active']:
				drawing['active'] = False
				clicked['end_x'] = actual_x
				clicked['end_y'] = actual_y

		cv2.namedWindow("Select Splash Zone", cv2.WINDOW_NORMAL)
		cv2.setMouseCallback("Select Splash Zone", on_mouse)
		print("\n🖱️  SPLASH ZONE SELECTION:")
		print("📍 Click and drag to create a bounding box around the splash detection area")
		print("� Recommended: Select area around water surface where splashes occur")
		print("⌨️  Press SPACE to confirm selection")
		print("⌨️  Press 'r' to retry/reset selection")
		print("⌨️  Press ESC to use default zone")

		last_state = None
		while True:
			# Only redraw if something changed
			current_state = (mouse_pos['x'], mouse_pos['y'], drawing['active'],
						   clicked['start_x'], clicked['start_y'], clicked['end_x'], clicked['end_y'])

			if mouse_pos['changed'] or last_state != current_state:
				display = clone.copy()

				# Show current mouse position crosshair
				if mouse_pos['x'] is not None and mouse_pos['y'] is not None:
					display_x = int(mouse_pos['x'] * display_scale) if display_scale < 1.0 else mouse_pos['x']
					display_y = int(mouse_pos['y'] * display_scale) if display_scale < 1.0 else mouse_pos['y']

					# Draw crosshair
					cv2.line(display, (display_x, 0), (display_x, display.shape[0]), (128, 128, 128), 1)
					cv2.line(display, (0, display_y), (display.shape[1], display_y), (128, 128, 128), 1)

				# Show selection rectangle
				if (clicked['start_x'] is not None and clicked['start_y'] is not None and
					clicked['end_x'] is not None and clicked['end_y'] is not None):

					# Calculate display coordinates
					start_x_disp = int(clicked['start_x'] * display_scale) if display_scale < 1.0 else clicked['start_x']
					start_y_disp = int(clicked['start_y'] * display_scale) if display_scale < 1.0 else clicked['start_y']
					end_x_disp = int(clicked['end_x'] * display_scale) if display_scale < 1.0 else clicked['end_x']
					end_y_disp = int(clicked['end_y'] * display_scale) if display_scale < 1.0 else clicked['end_y']

					# Ensure proper rectangle coordinates
					left = min(start_x_disp, end_x_disp)
					right = max(start_x_disp, end_x_disp)
					top = min(start_y_disp, end_y_disp)
					bottom = max(start_y_disp, end_y_disp)

					# Draw the splash zone rectangle with overlay
					overlay = display.copy()
					cv2.rectangle(overlay, (left, top), (right, bottom), (255, 0, 255), -1)
					cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
					cv2.rectangle(display, (left, top), (right, bottom), (255, 0, 255), 2)

					# Calculate actual dimensions in original frame coordinates
					actual_width = abs(clicked['end_x'] - clicked['start_x'])
					actual_height = abs(clicked['end_y'] - clicked['start_y'])

					# Add dimension labels
					cv2.putText(display, f"Splash Zone: {actual_width}x{actual_height}px",
							   (left, top-10 if top > 30 else bottom+25),
							   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

					# Show coordinates
					left_orig = min(clicked['start_x'], clicked['end_x'])
					right_orig = max(clicked['start_x'], clicked['end_x'])
					top_orig = min(clicked['start_y'], clicked['end_y'])
					bottom_orig = max(clicked['start_y'], clicked['end_y'])

					cv2.putText(display, f"({left_orig},{top_orig}) -> ({right_orig},{bottom_orig})",
							   (left, top-35 if top > 55 else bottom+50),
							   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

				# Show instructions
				cv2.putText(display, "Click & drag to select splash zone | SPACE=confirm | R=retry | ESC=default",
						   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
				cv2.putText(display, "Select area where water entry splashes occur",
						   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
				cv2.putText(display, f"Frame size: {w}x{h}",
						   (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

				cv2.imshow("Select Splash Zone", display)
				mouse_pos['changed'] = False
				last_state = current_state

			key = cv2.waitKey(30) & 0xFF

			if key == 32:  # SPACE to confirm
				if (clicked['start_x'] is not None and clicked['start_y'] is not None and
					clicked['end_x'] is not None and clicked['end_y'] is not None):

					# Calculate normalized coordinates
					left_norm = min(clicked['start_x'], clicked['end_x']) / w
					right_norm = max(clicked['start_x'], clicked['end_x']) / w
					top_norm = min(clicked['start_y'], clicked['end_y']) / h
					bottom_norm = max(clicked['start_y'], clicked['end_y']) / h

					# Validate selection size
					if (right_norm - left_norm) < 0.05 or (bottom_norm - top_norm) < 0.02:
						print("⚠️  Selection too small! Please select a larger area.")
						continue

					selection_complete = True
					break
				else:
					print("⚠️  No selection made! Please click and drag to select an area.")

			elif key == ord('r') or key == ord('R'):  # R to retry
				print("� Resetting selection...")
				clicked = {'start_x': None, 'start_y': None, 'end_x': None, 'end_y': None}
				drawing = {'active': False}

			elif key == 27:  # ESC for default
				print("⌨️  ESC pressed - using default splash zone")
				cv2.destroyAllWindows()
				return 0.7, 0.95, 0.0, 1.0  # Default: full width, bottom 25%

		cv2.destroyAllWindows()

		if selection_complete:
			# Show confirmation preview
			print("🔍 Showing preview of selected zone for 3 seconds...")
			preview_frame = frame.copy()

			# Calculate pixel coordinates
			left_px = int(left_norm * w)
			right_px = int(right_norm * w)
			top_px = int(top_norm * h)
			bottom_px = int(bottom_norm * h)

			# Draw zone with overlay
			overlay = preview_frame.copy()
			cv2.rectangle(overlay, (left_px, top_px), (right_px, bottom_px), (0, 255, 255), -1)
			cv2.addWeighted(overlay, 0.3, preview_frame, 0.7, 0, preview_frame)
			cv2.rectangle(preview_frame, (left_px, top_px), (right_px, bottom_px), (0, 255, 255), 3)

			# Add labels
			zone_width = right_px - left_px
			zone_height = bottom_px - top_px
			cv2.putText(preview_frame, f"SPLASH DETECTION ZONE", (left_px, top_px - 10),
					   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
			cv2.putText(preview_frame, f"Size: {zone_width}x{zone_height}px ({(right_norm-left_norm)*100:.1f}% x {(bottom_norm-top_norm)*100:.1f}%)",
					   (left_px, bottom_px + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
			cv2.putText(preview_frame, f"Coordinates: ({left_norm:.3f},{top_norm:.3f}) -> ({right_norm:.3f},{bottom_norm:.3f})",
					   (left_px, bottom_px + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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

			print(f"✅ Splash zone selected: ({left_px},{top_px}) -> ({right_px},{bottom_px}) pixels")
			print(f"📊 Normalized coordinates: ({left_norm:.3f},{top_norm:.3f}) -> ({right_norm:.3f},{bottom_norm:.3f})")
			print(f"📏 Zone size: {zone_width}x{zone_height}px ({(right_norm-left_norm)*100:.1f}% x {(bottom_norm-top_norm)*100:.1f}% of frame)")

			return top_norm, bottom_norm, left_norm, right_norm

	# This should never be reached, but just in case
	return 0.7, 0.95, 0.0, 1.0

def extract_dive_around_splash(video_path: str, splash_event: SplashEvent,
							  config: DetectionConfig, output_dir: str, dive_number: int,
							  progress_callback=None) -> str:
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

		# Draw splash detection zone (bounding box)
		zone_top = int(config.splash_zone_top_norm * h)
		zone_bottom = int(config.splash_zone_bottom_norm * h)
		zone_left = int(config.splash_zone_left_norm * w)
		zone_right = int(config.splash_zone_right_norm * w)

		# Draw bounding box rectangle
		cv2.rectangle(frame, (zone_left, zone_top), (zone_right, zone_bottom), (255, 0, 0), 2)
		cv2.putText(frame, "Splash Zone", (zone_left + 10, zone_top - 10),
				   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

		out.write(frame)
		current_frame += 1
		extracted_frames += 1

	cap.release()
	out.release()

	duration = extracted_frames / video_fps

	# Use progress callback to write success message instead of print
	success_msg = f"✅ Dive {dive_number + 1}: {output_filename} ({extracted_frames} frames, {duration:.1f}s)"
	if progress_callback:
		progress_callback.write(success_msg)
	else:
		print(success_msg)

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
  python splash_only_detector.py video.mp4 --auto-extraction-threshold --debug
  python splash_only_detector.py video.mp4 --zone 0.75 0.9 --threshold 15.0
  python splash_only_detector.py video.mp4 --bbox 0.7 0.9 0.2 0.8 --debug
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
					   default=[0.7, 0.95], help='Splash zone vertical coordinates (normalized 0.0-1.0)')
	parser.add_argument('--bbox', nargs=4, type=float, metavar=('TOP', 'BOTTOM', 'LEFT', 'RIGHT'),
					   help='Splash zone bounding box (normalized coordinates: top bottom left right)')
	parser.add_argument('--interactive-zone', action='store_true', default=True,
					   help='Use interactive zone selection with bounding box (click and drag) [DEFAULT]')
	parser.add_argument('--no-interactive-zone', action='store_true',
					   help='Disable interactive zone selection, use default zone or --zone/--bbox')

	# Threshold parameters
	parser.add_argument('--threshold', type=float, default=12.0, help='Base detection threshold')
	parser.add_argument('--adaptive-factor', type=float, default=1.2,
					   help='Adaptive threshold factor')
	parser.add_argument('--min-extraction-score', type=float, default=15.0,
					   help='Minimum score required to extract a dive (filters weak events)')
	parser.add_argument('--auto-extraction-threshold', action='store_true',
					   help='Automatically calculate optimal extraction threshold based on video content')
	parser.add_argument('--high-confidence-threshold', type=float, default=20.0,
					   help='Threshold for high confidence events')
	parser.add_argument('--no-close-dives', action='store_true',
					   help='Disable detection of dives closer than 5 seconds (default: allow close dives)')
	parser.add_argument('--allow-close-dives', action='store_true',
					   help='[DEPRECATED] Use --no-close-dives to disable. Close dives are allowed by default.')

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
	parser.add_argument('--pre-duration', type=float, default=6.0,
					   help='Seconds before splash to include')
	parser.add_argument('--post-duration', type=float, default=2.0,
					   help='Seconds after splash to include')

	# Debug options
	parser.add_argument('--debug', action='store_true', help='Enable debug plots')
	parser.add_argument('--debug-dir', default='debug_splash_detection',
					   help='Debug output directory')
	parser.add_argument('--no-extract', action='store_true',
					   help='Only detect, do not extract videos')
	parser.add_argument('--no-multi-pass', action='store_true',
					   help='Disable multi-pass hierarchical detection (default: multi-pass enabled)')
	parser.add_argument('--multi-pass', action='store_true',
					   help='[DEPRECATED] Use --no-multi-pass to disable. Multi-pass detection is enabled by default.')

	args = parser.parse_args()

	# Validate input
	if not os.path.exists(args.video_path):
		print(f"❌ Error: Video file not found: {args.video_path}")
		return 1

	# Handle zone selection
	if args.bbox:
		# Use provided bounding box coordinates (takes precedence)
		zone_top, zone_bottom, zone_left, zone_right = args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3]
		print(f"📦 Using provided bounding box: top={zone_top:.3f}, bottom={zone_bottom:.3f}, left={zone_left:.3f}, right={zone_right:.3f}")
	elif args.no_interactive_zone:
		# Use vertical zone with full width (interactive disabled)
		zone_top, zone_bottom = args.zone[0], args.zone[1]
		zone_left, zone_right = 0.0, 1.0  # Default to full width
		print(f"📊 Using vertical zone with full width: {zone_top:.3f} - {zone_bottom:.3f}")
	else:
		# Use interactive zone selection (DEFAULT behavior)
		try:
			print(f"🎯 Interactive zone selection enabled by default")
			zone_top, zone_bottom, zone_left, zone_right = get_splash_zone_interactive(args.video_path)
		except Exception as e:
			print(f"❌ Error during interactive zone selection: {e}")
			print("🔄 Falling back to default zone")
			zone_top, zone_bottom, zone_left, zone_right = 0.7, 0.95, 0.0, 1.0

	# Create configuration
	config = DetectionConfig(
		method=args.method,
		splash_zone_top_norm=zone_top,
		splash_zone_bottom_norm=zone_bottom,
		splash_zone_left_norm=zone_left,
		splash_zone_right_norm=zone_right,
		base_threshold=args.threshold,
		adaptive_threshold_factor=args.adaptive_factor,
		min_extraction_score=args.min_extraction_score,
		auto_extract_threshold=args.auto_extraction_threshold,
		high_confidence_threshold=args.high_confidence_threshold,
		medium_confidence_threshold=args.threshold,  # Use base threshold for medium
		allow_close_dives=not args.no_close_dives,  # Default True, disabled by --no-close-dives
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
	print(f"📊 Zone: vertical {config.splash_zone_top_norm:.3f}-{config.splash_zone_bottom_norm:.3f}, horizontal {config.splash_zone_left_norm:.3f}-{config.splash_zone_right_norm:.3f}")
	print(f"🎛️  Threshold: {config.base_threshold} (adaptive factor: {config.adaptive_threshold_factor})")
	print(f"🔍 Gaussian σ: {config.temporal_gaussian_sigma}, window: {config.temporal_window_size}")
	print(f"⛰️  Peak prominence: {config.min_peak_prominence}, distance: {config.min_peak_distance}")
	print(f"🔄 Multi-pass: {'✅ ENABLED' if not args.no_multi_pass else '❌ DISABLED'}")
	print(f"🤝 Close dives: {'✅ ALLOWED' if not args.no_close_dives else '❌ BLOCKED'}")

	if config.auto_extract_threshold:
		print(f"🤖 Filtering: AUTO-CALCULATE extraction threshold (intelligent mode)")
		print(f"   • Initial threshold: {config.min_extraction_score} (will be auto-adjusted)")
	else:
		print(f"✂️  Filtering: min extraction score = {config.min_extraction_score}, high confidence = {config.high_confidence_threshold}")

	print(f"⏱️  Extract: {config.pre_splash_duration}s before + {config.post_splash_duration}s after")

	try:
		# Initialize detector
		detector = SplashOnlyDetector(config)

		# Detect splash events
		start_time = time.time()
		if not args.no_multi_pass:  # Default True, disabled by --no-multi-pass
			print(f"🔄 Using multi-pass hierarchical detection to find farther/smaller dives")
			splash_events = detector.detect_splashes_in_video_multipass(args.video_path)
		else:
			splash_events = detector.detect_splashes_in_video(args.video_path)
		detection_time = time.time() - start_time

		# Generate debug plots after detection (multi-pass or single-pass)
		if config.enable_debug_plots:
			print("📊 Generating enhanced debug plots...")
			detector._generate_debug_plots()

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
			print(f"💡 Or check the splash zone selection and detection method")		# Extract dive videos
		if not args.no_extract and splash_events:
			print(f"\n🎬 EXTRACTING DIVE VIDEOS")
			print(f"=========================")

			os.makedirs(args.output_dir, exist_ok=True)

			extraction_start = time.time()
			extracted_paths = []

			# Use progress bar for extraction
			with tqdm(total=len(splash_events), desc="🎬 Extracting dives",
					 unit="videos", ncols=100,
					 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

				for i, event in enumerate(splash_events):
					try:
						# Update progress bar description with current dive info
						dive_desc = f"🎬 Extracting dive {i+1} (t={event.timestamp:.1f}s)"
						pbar.set_description(dive_desc)

						output_path = extract_dive_around_splash(
							args.video_path, event, config, args.output_dir, i, progress_callback=pbar
						)
						extracted_paths.append(output_path)

						# Update progress
						pbar.update(1)

					except Exception as e:
						pbar.write(f"❌ Failed to extract dive {i+1}: {e}")

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
