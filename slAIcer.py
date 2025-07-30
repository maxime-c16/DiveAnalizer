"""
DiveAnalyzer - Automated diving video analysis tool
Detects and extracts individual dives from swimming pool diving videos using computer vision.
"""

import os
import sys
import contextlib
import cv2
import numpy as np
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time
import json
import pickle
from datetime import datetime

# ==================== PERFORMANCE CACHE SYSTEM ====================
class PerformanceCache:
    """Manages performance statistics cache for dynamic progress estimation."""

    def __init__(self, cache_file="dive_performance_cache.pkl"):
        self.cache_file = cache_file
        self.stats = self.load_cache()

    def load_cache(self):
        """Load cached performance statistics."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass

        # Default statistics if no cache exists
        return {
            'detection_fps': 15.0,  # Average detection speed
            'extraction_time_per_frame': 0.13,  # Average seconds per frame extraction
            'total_runs': 0,
            'last_updated': None
        }

    def save_cache(self):
        """Save performance statistics to cache."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.stats, f)
        except Exception as e:
            print(f"Warning: Could not save performance cache: {e}")

    def update_stats(self, metrics):
        """Update performance statistics with new run data."""
        if not metrics:
            return

        # Update detection speed
        performance = metrics.get('performance', {})
        detection_fps = performance.get('detection_fps', 0)

        # Update extraction performance
        extraction_times = metrics.get('extraction_times', [])
        if extraction_times:
            total_extraction_time = sum(e['extraction_time'] for e in extraction_times)
            total_frames = sum(d['duration_frames'] for d in metrics.get('dive_durations', []))
            if total_frames > 0:
                extraction_time_per_frame = total_extraction_time / total_frames
            else:
                extraction_time_per_frame = self.stats['extraction_time_per_frame']
        else:
            extraction_time_per_frame = self.stats['extraction_time_per_frame']

        # Weighted average with previous stats (more weight to recent data)
        weight = 0.3 if self.stats['total_runs'] > 0 else 1.0

        if detection_fps > 0:
            self.stats['detection_fps'] = (1 - weight) * self.stats['detection_fps'] + weight * detection_fps

        self.stats['extraction_time_per_frame'] = (1 - weight) * self.stats['extraction_time_per_frame'] + weight * extraction_time_per_frame
        self.stats['total_runs'] += 1
        self.stats['last_updated'] = datetime.now().isoformat()

        self.save_cache()

class CustomProgressBar:
    """Custom progress bar without external dependencies."""

    def __init__(self, total, description="Progress", width=50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.1  # Update every 100ms

    def update(self, progress=1, description=None):
        """Update progress bar."""
        self.current += progress
        current_time = time.time()

        # Throttle updates to avoid excessive output
        if current_time - self.last_update < self.update_interval and self.current < self.total:
            return

        self.last_update = current_time

        if description:
            self.description = description

        self._draw()

    def set_total(self, new_total):
        """Update the total for dynamic progress tracking."""
        self.total = new_total
        self._draw()

    def _draw(self):
        """Draw the progress bar."""
        if self.total <= 0:
            return

        percent = min(100, (self.current / self.total) * 100)
        filled = int(self.width * self.current / self.total)
        bar = '█' * filled + '░' * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0 and self.current < self.total:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"

        # Clear line and print progress
        print(f"\r{self.description}: |{bar}| {percent:.1f}% ({self.current}/{self.total}) {eta_str}", end='', flush=True)

    def close(self):
        """Finish the progress bar."""
        self.current = self.total
        self._draw()
        print()  # New line

class DiveProgressTracker:
    """Tracks progress through detection and extraction phases with dynamic estimation."""

    def __init__(self, video_frames, video_fps, performance_cache):
        self.video_frames = video_frames
        self.video_fps = video_fps
        self.cache = performance_cache

        # Phase 1: Detection
        self.detection_fps = self.cache.stats['detection_fps']
        estimated_detection_time = video_frames / self.detection_fps
        self.detection_progress = CustomProgressBar(
            total=int(estimated_detection_time * 10),  # Update every 0.1s
            description="🔍 Detecting dives",
            width=40
        )

        # Phase 2: Extraction (initialized later)
        self.extraction_progress = None
        self.detected_dives = []
        self.extraction_started = False

        # Timing
        self.detection_start = time.time()
        self.extraction_start = None

    def update_detection_progress(self, frames_processed):
        """Update detection progress based on frames processed."""
        if not self.detection_progress:
            return

        elapsed = time.time() - self.detection_start
        expected_progress = int(elapsed * 10)  # Progress units (0.1s intervals)
        self.detection_progress.update(0, f"🔍 Detecting dives ({frames_processed}/{self.video_frames} frames)")

        # Update progress to match elapsed time
        if expected_progress > self.detection_progress.current:
            self.detection_progress.current = min(expected_progress, self.detection_progress.total)

    def detection_complete(self, detected_dives):
        """Mark detection phase complete and initialize extraction tracking."""
        if self.detection_progress:
            self.detection_progress.close()

        self.detected_dives = detected_dives

        if not detected_dives:
            print("🏊 No dives detected - skipping extraction phase")
            return

        # Calculate total extraction time estimate
        total_frames = sum(dive['duration_frames'] for dive in detected_dives)
        estimated_extraction_time = total_frames * self.cache.stats['extraction_time_per_frame']

        self.extraction_progress = CustomProgressBar(
            total=len(detected_dives),
            description=f"💾 Extracting {len(detected_dives)} dives",
            width=40
        )

        self.extraction_start = time.time()
        print(f"\n💾 Starting extraction of {len(detected_dives)} dives (estimated: {estimated_extraction_time:.1f}s)")

    def add_detected_dive(self, dive_info):
        """Add a detected dive to tracking."""
        self.detected_dives.append(dive_info)

    def dive_extraction_complete(self, dive_number):
        """Mark a dive extraction as complete."""
        if self.extraction_progress:
            remaining = len(self.detected_dives) - self.extraction_progress.current - 1
            self.extraction_progress.update(1, f"💾 Extracting dives ({remaining} remaining)")

    def extraction_complete(self):
        """Mark all extractions complete."""
        if self.extraction_progress:
            self.extraction_progress.close()

        extraction_time = time.time() - self.extraction_start if self.extraction_start else 0
        print(f"✅ All extractions completed in {extraction_time:.1f}s")

def write_compact_log(metrics, log_file="dive_analysis.log"):
    """Write compact machine-readable log for performance tracking."""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'video_info': metrics.get('video_info', {}),
            'performance': metrics.get('performance', {}),
            'dive_count': len(metrics.get('dive_durations', [])),
            'total_dive_frames': sum(d['duration_frames'] for d in metrics.get('dive_durations', [])),
            'extraction_times': metrics.get('extraction_times', []),
            'detection_time': metrics.get('detection_time', 0),
            'extraction_time': metrics.get('extraction_time', 0)
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    except Exception as e:
        print(f"Warning: Could not write compact log: {e}")

# ==================== MEDIAPIPE SETUP ====================

# ==================== DYNAMIC PROGRESS TRACKER ====================
class LegacyProgressTracker:
    """Manages dynamic progress tracking for dive detection and extraction."""

    def __init__(self, total_frames, cache):
        self.total_frames = total_frames
        self.cache = cache
        self.phase = "detection"  # "detection" or "extraction"

        # Phase 1: Detection progress
        self.detection_start_time = time.time()
        self.estimated_detection_time = cache.estimate_detection_time(total_frames)

        # Progress bars
        self.detection_pbar = None
        self.extraction_pbar = None

        # Dive tracking
        self.detected_dives = []
        self.completed_extractions = 0

        self.init_detection_progress()

    def init_detection_progress(self):
        """Initialize detection phase progress bar."""
        est_time_min = self.estimated_detection_time / 60
        # self.detection_pbar = tqdm(
        #     total=self.total_frames,
        #     desc=f"🔍 Detection (est. {est_time_min:.1f}min)",
        #     unit="frames",
        #     colour="blue",
        #     dynamic_ncols=True
        # )
        self.detection_pbar = None  # Placeholder

    def update_detection_progress(self, current_frame):
        """Update detection progress."""
        if self.detection_pbar:
            # Calculate dynamic rate based on actual performance
            elapsed = time.time() - self.detection_start_time
            if elapsed > 1:  # After 1 second, use real rate
                actual_fps = current_frame / elapsed
                remaining_frames = self.total_frames - current_frame
                est_remaining_time = remaining_frames / actual_fps if actual_fps > 0 else 0

                # Update description with real-time estimate
                est_min = est_remaining_time / 60
                desc = f"🔍 Detection ({actual_fps:.1f}fps, {est_min:.1f}min left)"
                self.detection_pbar.set_description(desc)

            self.detection_pbar.update(current_frame - self.detection_pbar.n)

    def add_detected_dive(self, dive_info):
        """Add a detected dive for extraction tracking."""
        self.detected_dives.append(dive_info)

    def finish_detection_phase(self):
        """Transition from detection to extraction phase."""
        if self.detection_pbar:
            self.detection_pbar.close()

        if not self.detected_dives:
            return

        # Calculate total extraction work
        total_dive_frames = sum(d['duration_frames'] for d in self.detected_dives)
        estimated_total_extraction = self.cache.estimate_extraction_time(total_dive_frames)

        self.phase = "extraction"
        est_time_min = estimated_total_extraction / 60

        # self.extraction_pbar = tqdm(
        #     total=len(self.detected_dives),
        #     desc=f"💾 Extraction (est. {est_time_min:.1f}min)",
        #     unit="dives",
        #     colour="green",
        #     dynamic_ncols=True
        # )
        self.extraction_pbar = None  # Placeholder

    def update_extraction_progress(self, completed_dive_info):
        """Update extraction progress when a dive completes."""
        if not self.extraction_pbar:
            return

        self.completed_extractions += 1

        # Calculate remaining work
        remaining_dives = len(self.detected_dives) - self.completed_extractions
        if remaining_dives > 0:
            # Estimate remaining time based on remaining dive frames
            remaining_frames = sum(
                d['duration_frames'] for d in self.detected_dives[self.completed_extractions:]
            )
            est_remaining_time = self.cache.estimate_extraction_time(remaining_frames)
            est_min = est_remaining_time / 60

            desc = f"💾 Extraction ({remaining_dives} left, {est_min:.1f}min)"
            self.extraction_pbar.set_description(desc)

        self.extraction_pbar.update(1)

    def finish_extraction_phase(self):
        """Finish extraction phase."""
        if self.extraction_pbar:
            self.extraction_pbar.close()

    def close(self):
        """Clean up progress bars."""
        if self.detection_pbar:
            self.detection_pbar.close()
        if self.extraction_pbar:
            self.extraction_pbar.close()

# ==================== LOGGING SUPPRESSION ====================
# Comprehensive log suppression for MediaPipe and TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # Suppress all TensorFlow logs
os.environ['GLOG_minloglevel'] = '3'         # Suppress Google logs (MediaPipe)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'    # Disable oneDNN optimization messages
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'    # Disable GPU to prevent GL context logs

class FilteredStderr:
    """Custom stderr filter to suppress specific MediaPipe/TensorFlow messages"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filtered_phrases = [
            'GL version:', 'INFO: Created TensorFlow', 'WARNING: All log messages before absl',
            'gl_context.cc:', 'TensorFlow Lite XNNPACK delegate', 'I0000', 'W0000',
            'renderer:', 'Intel(R)', 'INTEL-', 'OpenGL', 'Graphics'
        ]

    def write(self, text):
        if not any(phrase in text for phrase in self.filtered_phrases):
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

    def fileno(self):
        return self.original_stderr.fileno()

# Install the stderr filter
sys.stderr = FilteredStderr(sys.stderr)

# ==================== MEDIAPIPE INITIALIZATION ====================
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to completely suppress stderr at the file descriptor level"""
    original_stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        stderr_copy = os.dup(original_stderr_fd)
        try:
            os.dup2(devnull.fileno(), original_stderr_fd)
            yield
        finally:
            os.dup2(stderr_copy, original_stderr_fd)
            os.close(stderr_copy)

# Import MediaPipe with stderr suppressed
with suppress_stderr():
    import mediapipe as mp
    mp_pose = mp.solutions.pose

# ==================== VIDEO PROCESSING UTILITIES ====================

class ThreadedFrameProcessor:
	"""
	Threaded frame processing pipeline for faster video analysis.
	Decodes frames in background and preprocesses them for pose/splash detection.
	"""
	def __init__(self, video_path, target_fps=None, buffer_size=30):
		self.video_path = video_path
		self.target_fps = target_fps
		self.buffer_size = buffer_size
		self.frame_queue = queue.Queue(maxsize=buffer_size)
		self.processed_queue = queue.Queue(maxsize=buffer_size)
		self.stop_event = threading.Event()
		self.decode_thread = None
		self.process_thread = None

		# Get video info
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise ValueError(f"Could not open video file: {video_path}")
		self.video_fps = cap.get(cv2.CAP_PROP_FPS)
		if self.video_fps <= 0:
			self.video_fps = 30
		self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		cap.release()

		if target_fps is None:
			self.target_fps = self.video_fps
		self.interval = int(round(self.video_fps / self.target_fps))
		if self.interval == 0:
			self.interval = 1

	def _decode_frames(self):
		"""Background thread for frame decoding"""
		cap = cv2.VideoCapture(self.video_path)
		idx = 0
		saved_idx = 0

		try:
			while not self.stop_event.is_set():
				ret, frame = cap.read()
				if not ret:
					break

				if idx % self.interval == 0:
					try:
						self.frame_queue.put((saved_idx, frame), timeout=1.0)
						saved_idx += 1
					except queue.Full:
						if self.stop_event.is_set():
							break
						continue
				idx += 1
		finally:
			cap.release()
			# Signal end of frames
			try:
				self.frame_queue.put(None, timeout=1.0)
			except queue.Full:
				pass

	def _preprocess_frames(self):
		"""Background thread for frame preprocessing"""
		while not self.stop_event.is_set():
			try:
				item = self.frame_queue.get(timeout=1.0)
				if item is None:
					# End of frames signal
					self.processed_queue.put(None)
					break

				idx, frame = item
				h, w = frame.shape[:2]
				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				processed_data = {
					'idx': idx,
					'frame': frame,
					'rgb': rgb,
					'gray': gray,
					'shape': (h, w)
				}

				try:
					self.processed_queue.put(processed_data, timeout=1.0)
				except queue.Full:
					if self.stop_event.is_set():
						break
					continue

			except queue.Empty:
				continue

	def start(self):
		"""Start the processing pipeline"""
		self.stop_event.clear()
		self.decode_thread = threading.Thread(target=self._decode_frames, daemon=True)
		self.process_thread = threading.Thread(target=self._preprocess_frames, daemon=True)
		self.decode_thread.start()
		self.process_thread.start()

	def stop(self):
		"""Stop the processing pipeline"""
		self.stop_event.set()
		if self.decode_thread:
			self.decode_thread.join(timeout=2.0)
		if self.process_thread:
			self.process_thread.join(timeout=2.0)

	def get_frame(self, timeout=5.0):
		"""Get next processed frame"""
		try:
			return self.processed_queue.get(timeout=timeout)
		except queue.Empty:
			return None

	def __iter__(self):
		"""Iterator interface"""
		return self

	def __next__(self):
		item = self.get_frame()
		if item is None:
			raise StopIteration
		return item['idx'], item

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()

def frame_generator(video_path, target_fps=None):
	"""
	Yields (frame_index, frame) from a video file.
	If target_fps is None, uses the video's native framerate.
	If target_fps is specified, downsamples to that framerate.
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Could not open video file: {video_path}")

	video_fps = cap.get(cv2.CAP_PROP_FPS)
	if video_fps <= 0:
		video_fps = 30  # fallback

	# Use native framerate if target_fps not specified
	if target_fps is None:
		target_fps = video_fps

	interval = int(round(video_fps / target_fps))
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

def get_video_fps(video_path):
	"""Get the framerate of a video file."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		return 30  # fallback
	fps = cap.get(cv2.CAP_PROP_FPS)
	cap.release()
	return fps if fps > 0 else 30

class ThreadedPoseDetector:
	"""
	Improved threaded pose detection with shared pose instances.
	Maintains a pool of pose detectors to avoid recreation overhead.
	"""
	def __init__(self, max_workers=2):
		self.max_workers = max_workers
		self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
		# Create a pool of pose detectors (one per worker)
		self.pose_pool = queue.Queue()
		for _ in range(max_workers):
			with suppress_stderr():
				pose = mp_pose.Pose(static_image_mode=True)
				self.pose_pool.put(pose)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.shutdown()

	def shutdown(self):
		"""Clean up the thread pool executor and pose instances."""
		if self.executor:
			self.executor.shutdown(wait=True)
			self.executor = None
		# Clean up pose instances
		while not self.pose_pool.empty():
			try:
				pose = self.pose_pool.get_nowait()
				pose.close()
			except queue.Empty:
				break

	def _detect_pose_worker(self, rgb_image):
		"""
		Worker function that reuses pose detectors from the pool.
		"""
		pose = None
		try:
			# Get a pose detector from the pool
			pose = self.pose_pool.get(timeout=1.0)
			with suppress_stderr():
				result = pose.process(rgb_image)
			return result
		except queue.Empty:
			# Fallback: create temporary pose detector if pool is empty
			with suppress_stderr():
				temp_pose = mp_pose.Pose(static_image_mode=True)
				try:
					result = temp_pose.process(rgb_image)
					return result
				finally:
					temp_pose.close()
		finally:
			# Return pose detector to pool
			if pose is not None:
				try:
					self.pose_pool.put_nowait(pose)
				except queue.Full:
					# Pool is full, close this instance
					pose.close()

	def detect_pose_sync(self, rgb_image):
		"""
		Synchronous pose detection using pooled detectors.
		"""
		return self._detect_pose_worker(rgb_image)

	def detect_pose_async(self, rgb_image):
		"""
		Asynchronous pose detection using thread pool.
		Returns a Future object.
		"""
		return self.executor.submit(self._detect_pose_worker, rgb_image)

	def batch_detect_poses(self, frames_data, diver_zone_norm=None):
		"""
		Process multiple frames in parallel.
		Returns dictionary mapping frame_idx -> pose_landmarks
		"""
		if not frames_data:
			return {}

		futures = {}
		results = {}

		for frame_data in frames_data:
			frame_idx = frame_data['idx']
			rgb = frame_data['rgb']
			h, w = frame_data['shape']

			if diver_zone_norm is not None:
				# Extract ROI for pose detection
				left_norm, top_norm, right_norm, bottom_norm = diver_zone_norm
				left = int(left_norm * w)
				top = int(top_norm * h)
				right = int(right_norm * w)
				bottom = int(bottom_norm * h)
				roi = rgb[top:bottom, left:right]
				roi_bounds = (left, top, right, bottom, w, h)
			else:
				roi = rgb
				roi_bounds = None

			future = self.executor.submit(self.detect_pose_roi, roi, roi_bounds)
			futures[future] = (frame_idx, roi_bounds)

		# Collect results
		for future in as_completed(futures):
			frame_idx, roi_bounds = futures[future]
			try:
				pose_landmarks = future.result()
				results[frame_idx] = {
					'landmarks': pose_landmarks,
					'roi_bounds': roi_bounds,
					'diver_detected': pose_landmarks is not None
				}
			except Exception as e:
				print(f"Warning: Pose detection failed for frame {frame_idx}: {e}")
				results[frame_idx] = {
					'landmarks': None,
					'roi_bounds': roi_bounds,
					'diver_detected': False
				}

		return results

def auto_detect_board_y(frame):
	"""
	Automatically detect the likely diving board y position in the frame.
	Returns the y coordinate (int) of the detected board or None if not found.
	"""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (9,9), 0)
	edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
	lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=frame.shape[1]//5, maxLineGap=20)
	y_lines = []
	h, w = frame.shape[:2]
	if lines is not None:
		for l in lines:
			_, y1, _, y2 = l[0]
			# Only consider (nearly) horizontal lines away from the bottom
			if abs(y2 - y1) < 10 and min(y1,y2)>h//6 and max(y1,y2)<int(h*0.97):
				y_lines.append((y1 + y2) // 2)
		if y_lines:
			# The lowest horizontal except for the bottom of the image
			return int(sorted(y_lines)[-1])
	return None

# ==================== SPLASH DETECTION ALGORITHMS ====================
"""
Multiple splash detection methods available:
- frame_diff: Simple frame difference (fast, basic)
- optical_flow: Motion pattern analysis (medium complexity)
- contour: Foam/bubble detection (complex, HSV-based)
- motion_intensity: Gradient-based motion analysis with peak detection (recommended)
- combined: Voting system using multiple methods (robust but slow)

All methods return (is_splash: bool, confidence_score: float)
"""

def detect_splash_frame_diff(splash_band, prev_band, thresh=7.0):
	"""
	Current method: Simple frame difference
	"""
	if prev_band is None:
		return False, 0.0

	diff = cv2.absdiff(splash_band, prev_band)
	splash_score = np.mean(diff)
	return splash_score > thresh, splash_score

def detect_splash_optical_flow(splash_band, prev_band, flow_thresh=2.0, magnitude_thresh=10.0):
	"""
	Optical flow based splash detection
	Detects radial outward motion patterns characteristic of splashes
	"""
	if prev_band is None:
		return False, 0.0

	# Calculate dense optical flow using Farneback method
	flow = cv2.calcOpticalFlowFarneback(
		prev_band, splash_band, None,
		pyr_scale=0.5, levels=3, winsize=15, iterations=3,
		poly_n=5, poly_sigma=1.2, flags=0
	)

	if flow is None:
		return False, 0.0

	# Calculate flow magnitude and check for motion patterns
	flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
	mean_magnitude = np.mean(flow_magnitude)

	# Look for high magnitude flows (indicating rapid movement)
	high_flow_pixels = np.sum(flow_magnitude > magnitude_thresh)
	flow_density = high_flow_pixels / (splash_band.shape[0] * splash_band.shape[1])

	# Splash detected if significant motion with sufficient density
	is_splash = mean_magnitude > flow_thresh and flow_density > 0.1
	confidence = mean_magnitude * flow_density

	return is_splash, confidence

def detect_splash_contour_analysis(splash_band, prev_band, area_thresh=500, contour_thresh=3):
	"""
	Contour and foam analysis based splash detection
	Detects white foam and irregular shapes characteristic of water splashes
	"""
	if prev_band is None:
		return False, 0.0

	# Ensure splash_band is grayscale
	if len(splash_band.shape) == 3:
		splash_band = cv2.cvtColor(splash_band, cv2.COLOR_BGR2GRAY)

	# Convert to HSV for better water/foam separation
	splash_bgr = cv2.cvtColor(splash_band, cv2.COLOR_GRAY2BGR)
	splash_hsv = cv2.cvtColor(splash_bgr, cv2.COLOR_BGR2HSV)

	# Detect white/bright areas (foam)
	# High Value (brightness) and low Saturation typically indicate foam
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

def detect_splash_combined(splash_band, prev_band, splash_thresh=7.0):
	"""
	Combined splash detection using multiple methods with voting
	"""
	# Test all three methods
	diff_splash, diff_score = detect_splash_frame_diff(splash_band, prev_band, splash_thresh)

	# Optical flow (need colored version)
	flow_splash, flow_score = False, 0.0
	try:
		if prev_band is not None:
			# Create simple grid of points for optical flow
			h, w = splash_band.shape
			y, x = np.mgrid[10:h-10:20, 10:w-10:20].reshape(2, -1).astype(np.float32)
			points = np.column_stack([x, y]).reshape(-1, 1, 2)

			if len(points) > 0:
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
						flow_splash = mean_movement > 3.0
						flow_score = mean_movement
	except:
		pass

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

def detect_splash_motion_intensity(splash_band, prev_band, motion_thresh=12.0, consistency_thresh=0.15):
	"""
	Motion intensity based splash detection with temporal consistency
	Analyzes motion patterns and intensity over time for more robust detection
	Updated with more sensitive parameters for better detection
	"""
	if prev_band is None:
		return False, 0.0

	# Calculate frame difference
	diff = cv2.absdiff(splash_band, prev_band)

	# Apply Gaussian blur to reduce noise
	diff_blurred = cv2.GaussianBlur(diff, (3, 3), 0)

	# Calculate motion intensity using gradient magnitude
	grad_x = cv2.Sobel(diff_blurred, cv2.CV_64F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(diff_blurred, cv2.CV_64F, 0, 1, ksize=3)
	gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

	# Calculate statistics
	mean_intensity = np.mean(gradient_magnitude)
	std_intensity = np.std(gradient_magnitude)

	# High intensity areas (potential splash regions) - more sensitive threshold
	high_intensity_mask = gradient_magnitude > (mean_intensity + 1.5 * std_intensity)
	high_intensity_ratio = np.sum(high_intensity_mask) / gradient_magnitude.size

	# Adaptive threshold based on image statistics - more sensitive
	adaptive_thresh = max(motion_thresh, mean_intensity + 0.5 * std_intensity)

	# Splash detection criteria - more sensitive
	intensity_score = mean_intensity * (1 + 2 * high_intensity_ratio)
	spatial_consistency = high_intensity_ratio > consistency_thresh

	# Alternative detection: simple high mean intensity
	simple_detection = mean_intensity > motion_thresh

	is_splash = (intensity_score > adaptive_thresh and spatial_consistency) or simple_detection
	confidence = intensity_score * (1 + high_intensity_ratio)

	return is_splash, confidence

# ==================== DIVE DETECTION ENGINE ====================
def find_next_dive_threaded(frame_gen, board_y_norm, water_y_norm=0.95,
                           splash_zone_top_norm=None, splash_zone_bottom_norm=None,
                           diver_zone_norm=None, start_search_at_idx=0, debug=False, splash_method='combined',
                           video_fps=30):
	"""
	Improved threaded version of dive detection with batched processing.
	Processes frames in batches to reduce threading overhead.
	"""
	print("🚀 Using improved threaded dive detection for better performance...")

	# Use batched processing to reduce threading overhead
	batch_size = 8  # Process 8 frames at a time
	pose_detector = ThreadedPoseDetector(max_workers=3)

	try:
		# Dive states
		WAITING = 0
		DIVER_ON_PLATFORM = 1
		DIVING = 2

		state = WAITING
		start_idx = None
		end_idx = None
		confidence = 'high'

		# Dynamic timing constraints based on video framerate
		min_dive_frames = int(1.0 * video_fps)
		max_no_splash_frames = int(6.0 * video_fps)
		frames_since_start = 0
		frames_without_diver = 0

		print(f"Using dynamic timing: min_dive={min_dive_frames} frames, max_timeout={max_no_splash_frames} frames at {video_fps:.1f}fps")

		# For splash detection
		prev_gray = None
		splash_detected_idx = None
		splash_thresh = 12.0

		# Peak-based splash event detection for motion_intensity
		splash_event_state = 'waiting'
		peak_score = 0
		frames_since_peak = 0
		cooldown_frames = 5

		# Diver detection in zone
		consecutive_diver_frames = 0
		consecutive_no_diver_frames = 0
		diver_detection_threshold = 3

		# Batch processing variables
		frame_batch = []
		pose_futures = []

		# Process frames with batched pose detection
		for idx, frame in frame_gen:
			h, w = frame.shape[:2]
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Add frame to batch for pose detection
			if diver_zone_norm is not None:
				left_norm, top_norm, right_norm, bottom_norm = diver_zone_norm
				left = int(left_norm * w)
				top = int(top_norm * h)
				right = int(right_norm * w)
				bottom = int(bottom_norm * h)
				roi = rgb[top:bottom, left:right]
				if roi.size > 0:
					# Submit pose detection asynchronously
					future = pose_detector.detect_pose_async(roi)
					pose_futures.append((idx, future, (left, top, right, bottom, w, h)))
				else:
					pose_futures.append((idx, None, None))
			else:
				# Full frame pose detection
				future = pose_detector.detect_pose_async(rgb)
				pose_futures.append((idx, future, None))

			# Process completed pose detections
			diver_in_zone = False
			pose_landmarks_full_frame = None

			# Check if we have a result for this frame
			if len(pose_futures) > 0:
				current_idx, current_future, roi_info = pose_futures[0]
				if current_idx == idx and current_future is not None:
					try:
						# Get pose result (this might block briefly)
						pose_result = current_future.result(timeout=0.001)  # Very short timeout
						if pose_result and pose_result.pose_landmarks:
							diver_in_zone = True
							if roi_info:
								# Convert ROI landmarks back to full frame coordinates
								left, top, right, bottom, frame_w, frame_h = roi_info
								pose_landmarks_full_frame = []
								for landmark in pose_result.pose_landmarks.landmark:
									full_x = (landmark.x * (right - left) + left) / frame_w
									full_y = (landmark.y * (bottom - top) + top) / frame_h
									pose_landmarks_full_frame.append((full_x, full_y, landmark.z))
							else:
								pose_landmarks_full_frame = [(lm.x, lm.y, lm.z) for lm in pose_result.pose_landmarks.landmark]
						pose_futures.pop(0)  # Remove processed item
					except:
						# Pose detection not ready yet, use previous state or default
						pose_futures.pop(0)  # Remove failed item
						pass

			# --- Splash detection (same as before) ---
			if splash_zone_top_norm is not None and splash_zone_bottom_norm is not None:
				band_top = int(splash_zone_top_norm * h)
				band_bot = int(splash_zone_bottom_norm * h)
			else:
				water_y_px = int(water_y_norm * h)
				splash_band_height = 70
				band_bot = water_y_px
				band_top = max(water_y_px - splash_band_height, 0)

			splash_band = gray[band_top:band_bot, :]
			splash_score = 0
			splash_this_frame = False

			if prev_gray is not None:
				prev_band = prev_gray[band_top:band_bot, :]

				if splash_method == 'motion_intensity':
					_, splash_score = detect_splash_motion_intensity(splash_band, prev_band)
					splash_this_frame = False

					if splash_event_state == 'waiting':
						if splash_score > splash_thresh:
							splash_event_state = 'in_splash'
							peak_score = splash_score

					elif splash_event_state == 'in_splash':
						if splash_score > peak_score:
							peak_score = splash_score
							frames_since_peak = 0
						else:
							frames_since_peak += 1

						if splash_score <= splash_thresh:
							splash_event_state = 'post_splash'
							splash_this_frame = True
							frames_since_peak = 0

					elif splash_event_state == 'post_splash':
						frames_since_peak += 1
						if frames_since_peak >= cooldown_frames:
							splash_event_state = 'waiting'

				elif splash_method == 'combined':
					splash_this_frame, splash_score, _ = detect_splash_combined(splash_band, prev_band, splash_thresh)

				if splash_this_frame:
					splash_detected_idx = idx
			prev_gray = gray

			# Update diver presence counters
			if diver_in_zone:
				consecutive_diver_frames += 1
				consecutive_no_diver_frames = 0
			else:
				consecutive_diver_frames = 0
				consecutive_no_diver_frames += 1

			# --- State Machine Logic (same as before) ---
			if state == WAITING:
				if consecutive_diver_frames >= diver_detection_threshold:
					state = DIVER_ON_PLATFORM
					start_idx = idx - consecutive_diver_frames + 1
					frames_since_start = consecutive_diver_frames - 1

			elif state == DIVER_ON_PLATFORM:
				frames_since_start += 1
				if consecutive_no_diver_frames >= diver_detection_threshold:
					state = DIVING
					frames_without_diver = consecutive_no_diver_frames

			elif state == DIVING:
				frames_since_start += 1
				if not diver_in_zone:
					frames_without_diver += 1
				else:
					frames_without_diver = 0

				dive_long_enough = frames_since_start >= min_dive_frames

				if dive_long_enough:
					if splash_this_frame or (splash_detected_idx is not None and abs(idx - splash_detected_idx) <= 3):
						if frames_without_diver >= diver_detection_threshold:
							end_idx = idx
							confidence = 'high'
							break
					elif frames_without_diver >= max_no_splash_frames:
						end_idx = idx
						confidence = 'low'
						break

	finally:
		# Clean up any remaining futures
		for _, future, _ in pose_futures:
			if future is not None:
				try:
					future.cancel()
				except:
					pass
		pose_detector.shutdown()

	return start_idx, end_idx, confidence

def find_next_dive(frame_gen, board_y_norm, water_y_norm=0.95,
				   splash_zone_top_norm=None, splash_zone_bottom_norm=None,
				   diver_zone_norm=None, start_search_at_idx=0, debug=False, splash_method='combined',
				   video_fps=30, use_threading=True, enable_pose_optimization=False):
	"""
	Core dive detection algorithm using state machine approach.
	Conservative optimization: Reduce pose detection only when safe (during confirmed diving state).
	"""

	# Threading optimization - reuse pose detector + optional selective detection
	if use_threading:
		if enable_pose_optimization:
			print("🚀 Using conservative optimized pose detection (selective + reused detector)...")
		else:
			print("🚀 Using basic optimized pose detection (reused detector only)...")
		with suppress_stderr():
			pose = mp_pose.Pose(static_image_mode=True)
	else:
		pose = None

	# Dive states
	WAITING = 0
	DIVER_ON_PLATFORM = 1
	DIVING = 2

	state = WAITING
	start_idx = None
	end_idx = None
	confidence = 'high'

	# Dynamic timing constraints based on video framerate
	min_dive_frames = int(1.0 * video_fps)
	max_no_splash_frames = int(6.0 * video_fps)
	frames_since_start = 0
	frames_without_diver = 0

	print(f"Using dynamic timing: min_dive={min_dive_frames} frames, max_timeout={max_no_splash_frames} frames at {video_fps:.1f}fps")

	# Adaptive pose detection optimization
	last_pose_detection_frame = -1
	pose_cooldown_frames = int(0.2 * video_fps)  # Skip ~0.2s worth of frames
	pose_detections_skipped = 0
	total_pose_opportunities = 0

	# Debug window management
	skipping_to_action = debug
	debug_frame_delay = max(1, int(1000 / video_fps))

	# For splash detection
	prev_gray = None
	splash_detected_idx = None
	splash_thresh = 12.0

	# Peak-based splash event detection for motion_intensity
	splash_event_state = 'waiting'
	peak_score = 0
	frames_since_peak = 0
	cooldown_frames = 5

	# Diver detection in zone
	consecutive_diver_frames = 0
	consecutive_no_diver_frames = 0
	diver_detection_threshold = 3

	for idx, frame in frame_gen:
		h, w = frame.shape[:2]
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# --- Splash detection setup ---
		if splash_zone_top_norm is not None and splash_zone_bottom_norm is not None:
			band_top = int(splash_zone_top_norm * h)
			band_bot = int(splash_zone_bottom_norm * h)
		else:
			water_y_px = int(water_y_norm * h)
			splash_band_height = 70
			band_bot = water_y_px
			band_top = max(water_y_px - splash_band_height, 0)

		# Calculate splash score using selected method
		splash_band = gray[band_top:band_bot, :]
		splash_score = 0
		splash_this_frame = False

		if prev_gray is not None:
			prev_band = prev_gray[band_top:band_bot, :]

			if splash_method == 'motion_intensity':
				_, splash_score = detect_splash_motion_intensity(splash_band, prev_band)
				splash_this_frame = False

				if splash_event_state == 'waiting':
					if splash_score > splash_thresh:
						splash_event_state = 'in_splash'
						peak_score = splash_score

				elif splash_event_state == 'in_splash':
					if splash_score > peak_score:
						peak_score = splash_score
						frames_since_peak = 0
					else:
						frames_since_peak += 1

					if splash_score <= splash_thresh:
						splash_event_state = 'post_splash'
						splash_this_frame = True
						frames_since_peak = 0

				elif splash_event_state == 'post_splash':
					frames_since_peak += 1
					if frames_since_peak >= cooldown_frames:
						splash_event_state = 'waiting'

			elif splash_method == 'combined':
				splash_this_frame, splash_score, _ = detect_splash_combined(splash_band, prev_band, splash_thresh)

			if splash_this_frame:
				splash_detected_idx = idx
		prev_gray = gray

		# --- Conservative Adaptive Diver detection in zone ---
		diver_in_zone = False
		pose_landmarks_full_frame = None

		# Conservative adaptive pose detection - only skip when we're very confident
		should_detect_pose = True  # Default to always detect
		total_pose_opportunities += 1

		# Only apply optimization if enabled and we're in a safe state to skip
		if enable_pose_optimization and state == DIVING and frames_without_diver > diver_detection_threshold * 2:
			# Diver has been gone for a while during diving - safe to skip occasionally
			frames_since_last_detection = idx - last_pose_detection_frame
			if frames_since_last_detection < pose_cooldown_frames:
				should_detect_pose = False
				diver_in_zone = False  # Safe assumption during diving cooldown

		# Perform pose detection when needed
		if should_detect_pose:
			if diver_zone_norm is not None:
				left_norm, top_norm, right_norm, bottom_norm = diver_zone_norm
				left = int(left_norm * w)
				top = int(top_norm * h)
				right = int(right_norm * w)
				bottom = int(bottom_norm * h)

				roi = rgb[top:bottom, left:right]
				if roi.size > 0:
					if use_threading and pose is not None:
						# Use reused pose detector
						res = pose.process(roi)
					else:
						# Create temporary pose detector
						with suppress_stderr():
							temp_pose = mp_pose.Pose(static_image_mode=True)
							res = temp_pose.process(roi)
							temp_pose.close()

					if res.pose_landmarks:
						diver_in_zone = True
						pose_landmarks_full_frame = []
						for landmark in res.pose_landmarks.landmark:
							full_x = (landmark.x * (right - left) + left) / w
							full_y = (landmark.y * (bottom - top) + top) / h
							pose_landmarks_full_frame.append((full_x, full_y, landmark.z))

					last_pose_detection_frame = idx  # Mark that we performed detection
			else:
				# Fallback: detect pose in entire frame (old behavior)
				if use_threading and pose is not None:
					# Use reused pose detector
					res = pose.process(rgb)
				else:
					# Create temporary pose detector
					with suppress_stderr():
						temp_pose = mp_pose.Pose(static_image_mode=True)
						res = temp_pose.process(rgb)
						temp_pose.close()

				diver_in_zone = res.pose_landmarks is not None
				if res.pose_landmarks:
					pose_landmarks_full_frame = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]

				last_pose_detection_frame = idx  # Mark that we performed detection
		else:
			# Pose detection was skipped due to optimization
			pose_detections_skipped += 1

		# Update diver presence counters
		if diver_in_zone:
			consecutive_diver_frames += 1
			consecutive_no_diver_frames = 0
		else:
			consecutive_diver_frames = 0
			consecutive_no_diver_frames += 1

		# --- State Machine Logic ---
		if state == WAITING:
			if consecutive_diver_frames >= diver_detection_threshold:
				state = DIVER_ON_PLATFORM
				start_idx = idx - consecutive_diver_frames + 1
				frames_since_start = consecutive_diver_frames - 1

		elif state == DIVER_ON_PLATFORM:
			frames_since_start += 1
			if consecutive_no_diver_frames >= diver_detection_threshold:
				state = DIVING
				frames_without_diver = consecutive_no_diver_frames

		elif state == DIVING:
			frames_since_start += 1
			if not diver_in_zone:
				frames_without_diver += 1
			else:
				frames_without_diver = 0

			dive_long_enough = frames_since_start >= min_dive_frames

			if dive_long_enough:
				if splash_this_frame or (splash_detected_idx is not None and abs(idx - splash_detected_idx) <= 3):
					if frames_without_diver >= diver_detection_threshold:
						end_idx = idx
						confidence = 'high'
						break
				elif frames_without_diver >= max_no_splash_frames:
					end_idx = idx
					confidence = 'low'
					break

	# Cleanup and performance reporting
	if use_threading and pose is not None:
		pose.close()

		# Report optimization performance
		if enable_pose_optimization and total_pose_opportunities > 0:
			skip_percentage = (pose_detections_skipped / total_pose_opportunities) * 100
			print(f"⚡ Conservative pose optimization: skipped {pose_detections_skipped}/{total_pose_opportunities} operations ({skip_percentage:.1f}% reduction)")
		elif not enable_pose_optimization:
			print("⚡ Pose optimization disabled - using full detection")
		else:
			print("⚡ Conservative optimization: full pose detection used (no safe skip opportunities)")
	if debug:
		cv2.destroyAllWindows()

	# Adjust frame indices to account for the start_search_at_idx offset
	if start_idx is not None:
		start_idx += start_search_at_idx
	if end_idx is not None:
		end_idx += start_search_at_idx

	return start_idx, end_idx, confidence

def print_detailed_metrics(metrics, output_dir):
	"""Print comprehensive metrics about the dive analysis performance."""
	print("\n" + "="*80)
	print("📊 DIVE ANALYSIS METRICS")
	print("="*80)

	# Video Information
	video_info = metrics.get('video_info', {})
	print(f"🎥 Video Information:")
	print(f"    📁 File: {video_info.get('filename', 'N/A')}")
	print(f"    📏 Resolution: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}")
	print(f"    🎬 FPS: {video_info.get('fps', 'N/A'):.1f}")
	print(f"    ⏱️  Duration: {video_info.get('duration_seconds', 'N/A'):.1f}s ({video_info.get('total_frames', 'N/A')} frames)")
	print(f"    💾 File Size: {video_info.get('file_size_mb', 'N/A'):.1f} MB")

	# Processing Performance
	print(f"\n⚡ Processing Performance:")
	print(f"    🔍 Detection Time: {metrics.get('detection_time', 0):.2f}s")
	print(f"    💾 Extraction Time: {metrics.get('extraction_time', 0):.2f}s")
	print(f"    ⏱️  Total Processing: {metrics.get('total_processing_time', 0):.2f}s")
	print(f"    🖼️  Frames Processed: {metrics.get('total_frames_processed', 'N/A')}")

	performance = metrics.get('performance', {})
	if performance:
		print(f"    🚀 Detection Speed: {performance.get('detection_fps', 0):.1f} fps")
		print(f"    ⚡ Realtime Ratio: {performance.get('realtime_ratio', 0):.2f}x")
		print(f"    📈 Efficiency: {performance.get('efficiency_rating', 'N/A')}")

	# Dive Statistics
	dive_count = len(metrics.get('dive_durations', []))
	print(f"\n🏊 Dive Statistics:")
	print(f"    📊 Total Dives Found: {dive_count}")

	if dive_count > 0:
		dive_stats = metrics.get('dive_stats', {})
		print(f"    ⏱️  Average Duration: {dive_stats.get('average_duration', 0):.2f}s")
		print(f"    🏆 Longest Dive: #{dive_stats.get('longest_dive', {}).get('dive_number', 'N/A')} ({dive_stats.get('longest_dive', {}).get('duration_seconds', 0):.2f}s)")
		print(f"    ⚡ Shortest Dive: #{dive_stats.get('shortest_dive', {}).get('dive_number', 'N/A')} ({dive_stats.get('shortest_dive', {}).get('duration_seconds', 0):.2f}s)")
		print(f"    🏊 Total Dive Time: {dive_stats.get('total_dive_time', 0):.2f}s")

		# Individual dive details
		print(f"\n    📝 Individual Dive Details:")
		for dive_info in metrics.get('dive_durations', []):
			print(f"       🏊 Dive #{dive_info['dive_number']}: {dive_info['duration_seconds']:.2f}s ({dive_info['duration_frames']} frames)")

	# Extraction Performance
	extraction_times = metrics.get('extraction_times', [])
	if extraction_times:
		print(f"\n💾 Extraction Performance:")
		avg_extraction = sum(e['extraction_time'] for e in extraction_times) / len(extraction_times)
		print(f"    ⚡ Average Extraction: {avg_extraction:.2f}s")
		print(f"    📊 Extraction Details:")
		for ext_info in extraction_times:
			print(f"       💾 Dive #{ext_info['dive_number']}: {ext_info['extraction_time']:.2f}s")

	# Output Information
	print(f"\n📁 Output:")
	print(f"    📂 Directory: {output_dir}")
	print(f"    📄 Files Created: {dive_count} dive video(s)")

	print("="*80)


def save_compact_performance_log(metrics, output_dir):
	"""Save a compact, machine-readable performance log for future reference."""
	log_data = {
		'timestamp': datetime.now().isoformat(),
		'video_info': metrics.get('video_info', {}),
		'performance': {
			'detection_fps': metrics.get('performance', {}).get('detection_fps', 0),
			'detection_time': metrics.get('detection_time', 0),
			'total_frames_processed': metrics.get('total_frames_processed', 0),
			'total_processing_time': metrics.get('total_processing_time', 0)
		},
		'dive_stats': {
			'total_dives': len(metrics.get('dive_durations', [])),
			'total_dive_frames': sum(d['duration_frames'] for d in metrics.get('dive_durations', [])),
			'total_extraction_time': sum(e['extraction_time'] for e in metrics.get('extraction_times', [])),
			'avg_extraction_per_frame': 0
		}
	}

	# Calculate extraction time per frame
	total_frames = log_data['dive_stats']['total_dive_frames']
	total_extraction = log_data['dive_stats']['total_extraction_time']
	if total_frames > 0:
		log_data['dive_stats']['avg_extraction_per_frame'] = total_extraction / total_frames

	# Save to output directory
	if output_dir:
		log_file = os.path.join(output_dir, "performance_log.json")
		try:
			with open(log_file, 'w') as f:
				json.dump(log_data, f, indent=2)
			print(f"📄 Compact performance log saved: {log_file}")
		except Exception as e:
			print(f"Warning: Could not save performance log: {e}")


def detect_and_extract_dives_realtime(video_path, board_y_norm, water_y_norm,
									  splash_zone_top_norm=None, splash_zone_bottom_norm=None,
									  diver_zone_norm=None, debug=False, splash_method='motion_intensity',
									  use_threading=True, enable_pose_optimization=True, output_dir=None,
									  show_pose_overlay=False, preserve_audio=True):
	"""
	Real-time dive detection and extraction: spawns background threads immediately when dives are detected.
	This allows extraction to happen in parallel with continued detection for maximum efficiency.

	Returns:
		Dictionary containing dives list and comprehensive metrics
	"""
	import threading
	import queue
	import time
	import os
	from concurrent.futures import ThreadPoolExecutor, as_completed

	# 📊 PERFORMANCE CACHE AND PROGRESS TRACKING
	performance_cache = PerformanceCache()

	# 📊 METRICS TRACKING INITIALIZATION
	start_time = time.time()
	metrics = {
		'start_time': start_time,
		'detection_start_time': None,
		'detection_end_time': None,
		'extraction_end_time': None,
		'total_processing_time': 0,
		'detection_time': 0,
		'extraction_time': 0,
		'total_frames_processed': 0,
		'video_info': {},
		'dives_detected': 0,
		'high_confidence_dives': 0,
		'low_confidence_dives': 0,
		'dive_durations': [],
		'extraction_times': [],
		'performance': {},
		'settings': {
			'threading_used': use_threading,
			'optimization_used': enable_pose_optimization,
			'splash_method': splash_method
		}
	}

	# Get comprehensive video info
	video_fps = get_video_fps(video_path)
	cap = cv2.VideoCapture(video_path)
	total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	video_duration_seconds = total_video_frames / video_fps if video_fps > 0 else 0
	video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	cap.release()

	metrics['video_info'] = {
		'filename': os.path.basename(video_path),
		'fps': video_fps,
		'total_frames': total_video_frames,
		'duration_seconds': video_duration_seconds,
		'width': video_width,
		'height': video_height,
		'file_size_mb': os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0
	}
	threading_text = "with threading" if use_threading else "without threading"
	print(f"🚀 Real-time dive detection and extraction at {video_fps:.1f}fps {threading_text}")
	print(f"📊 Video: {video_width}x{video_height}, {total_video_frames} frames, {video_duration_seconds:.1f}s, {metrics['video_info']['file_size_mb']:.1f}MB")

	if output_dir:
		print(f"📁 Extraction output directory: {output_dir}")

	# Initialize progress tracker
	progress_tracker = DiveProgressTracker(total_video_frames, video_fps, performance_cache)
	print(f"📈 Using cached performance data from {performance_cache.stats['total_runs']} previous runs")

	dives = []
	start_search_at_idx = 0
	extraction_futures = []
	dive_counter = 0

	# Create thread pool for immediate extractions
	max_extraction_workers = min(4, os.cpu_count() or 2)
	executor = ThreadPoolExecutor(max_workers=max_extraction_workers)
	print(f"🎬 Using {max_extraction_workers} background extraction threads")

	# Start detection timing
	metrics['detection_start_time'] = time.time()

	try:
		while True:
			frame_gen = frame_generator(video_path)
			for _ in range(start_search_at_idx):
				try:
					next(frame_gen)
				except StopIteration:
					frame_gen = None
					break

			if frame_gen is None:
				print("  -> No more dives found in the remainder of the video.")
				break

			print(f"🔍 Searching for dive from frame {start_search_at_idx}...")

			# Update detection progress
			progress_tracker.update_detection_progress(start_search_at_idx)

			start_idx, end_idx, confidence = find_next_dive(
				frame_gen, board_y_norm, water_y_norm,
				splash_zone_top_norm, splash_zone_bottom_norm, diver_zone_norm,
				start_search_at_idx=start_search_at_idx, debug=debug, splash_method=splash_method,
				video_fps=video_fps, use_threading=use_threading, enable_pose_optimization=enable_pose_optimization
			)

			if start_idx is not None and end_idx is not None:
				min_dive_frames = int(1.0 * video_fps)  # Dynamic minimum based on framerate
				if (end_idx - start_idx) >= min_dive_frames:
					confidence_text = f"({confidence} confidence)" if confidence == 'low' else ""
					print(f"  ✅ Found dive {dive_counter + 1}: Start {start_idx}, End {end_idx} {confidence_text}")

					# 📊 Calculate dive metrics
					dive_duration_frames = end_idx - start_idx
					dive_duration_seconds = dive_duration_frames / video_fps
					dive_start_time = start_idx / video_fps

					dives.append((start_idx, end_idx, confidence))
					metrics['dives_detected'] += 1
					dive_info = {
						'dive_number': dive_counter + 1,
						'start_frame': start_idx,
						'end_frame': end_idx,
						'duration_frames': dive_duration_frames,
						'duration_seconds': dive_duration_seconds,
						'start_time_seconds': dive_start_time,
						'confidence': confidence
					}
					metrics['dive_durations'].append(dive_info)

					# Add to progress tracker
					progress_tracker.add_detected_dive(dive_info)

					if confidence == 'high':
						metrics['high_confidence_dives'] += 1
					else:
						metrics['low_confidence_dives'] += 1

					# 🚀 IMMEDIATELY SPAWN BACKGROUND EXTRACTION THREAD
					if output_dir:
						print(f"    🎬 Starting background extraction for dive {dive_counter + 1}...")
						future = executor.submit(
							extract_and_save_dive,
							video_path, dive_counter, start_idx, end_idx, confidence,
							output_dir, diver_zone_norm, video_fps=video_fps,
							show_pose_overlay=show_pose_overlay, preserve_audio=preserve_audio
						)
						extraction_futures.append((future, dive_counter + 1))

					dive_counter += 1
					start_search_at_idx = end_idx + 1
				else:
					print(f"  ⚠️  Dive ignored (too short): Start {start_idx}, End {end_idx}")
					start_search_at_idx = end_idx + 1
			else:
				print("  -> No more dives found in the remainder of the video.")
				break

		# Mark end of detection phase
		metrics['detection_end_time'] = time.time()
		metrics['detection_time'] = metrics['detection_end_time'] - metrics['detection_start_time']
		metrics['total_frames_processed'] = start_search_at_idx

		# Transition to extraction phase
		progress_tracker.detection_complete(metrics['dive_durations'])

		# Wait for all background extractions to complete
		if extraction_futures:
			print(f"\n⏳ Waiting for {len(extraction_futures)} background extractions to complete...")
			completed = 0

			for future, dive_num in extraction_futures:
				try:
					extraction_duration = future.result()  # This now returns the actual extraction time
					completed += 1
					print(f"    ✅ Dive {dive_num} extraction completed in {extraction_duration:.1f}s ({completed}/{len(extraction_futures)})")

					# Update progress tracker
					progress_tracker.dive_extraction_complete(dive_num)

					# Track extraction timing
					metrics['extraction_times'].append({
						'dive_number': dive_num,
						'extraction_time': extraction_duration
					})

				except Exception as e:
					print(f"    ❌ Dive {dive_num} extraction failed: {e}")

			print(f"🎉 All extractions completed!")
			progress_tracker.extraction_complete()

		# Mark end of extraction phase
		metrics['extraction_end_time'] = time.time()
		metrics['extraction_time'] = metrics['extraction_end_time'] - metrics['detection_end_time'] if metrics['detection_end_time'] else 0

	finally:
		executor.shutdown(wait=True)

	# 📊 FINAL METRICS CALCULATION
	end_time = time.time()
	metrics['total_processing_time'] = end_time - start_time

	# Calculate performance metrics
	if metrics['detection_time'] > 0:
		detection_fps = metrics['total_frames_processed'] / metrics['detection_time']
		realtime_ratio = detection_fps / video_fps if video_fps > 0 else 0

		metrics['performance'] = {
			'detection_fps': detection_fps,
			'realtime_ratio': realtime_ratio,
			'efficiency_rating': 'Excellent' if realtime_ratio >= 1.0 else 'Good' if realtime_ratio >= 0.8 else 'Needs Improvement'
		}

	# Calculate dive statistics
	if metrics['dive_durations']:
		durations = [d['duration_seconds'] for d in metrics['dive_durations']]
		metrics['dive_stats'] = {
			'longest_dive': max(metrics['dive_durations'], key=lambda x: x['duration_seconds']),
			'shortest_dive': min(metrics['dive_durations'], key=lambda x: x['duration_seconds']),
			'average_duration': sum(durations) / len(durations),
			'total_dive_time': sum(durations)
		}

	# Update performance cache for future runs
	performance_cache.update_stats(metrics)

	# Save compact performance log
	write_compact_log(metrics, "dive_analysis.log")

	# Print comprehensive metrics
	print_detailed_metrics(metrics, output_dir)

	return {'dives': dives, 'metrics': metrics}
def detect_all_dives(video_path, board_y_norm, water_y_norm,
					 splash_zone_top_norm=None, splash_zone_bottom_norm=None,
					 diver_zone_norm=None, debug=False, splash_method='motion_intensity',
					 use_threading=True, enable_pose_optimization=False):
	"""
	Legacy function: Detects all dive sequences sequentially (detection first, then extraction).
	For real-time detection+extraction, use detect_and_extract_dives_realtime() instead.
	Returns a list of (start_idx, end_idx, confidence) tuples for each dive.
	"""

	# Get video framerate for dynamic timing
	video_fps = get_video_fps(video_path)
	threading_text = "with threading" if use_threading else "without threading"
	print(f"Analyzing video at native framerate: {video_fps:.1f}fps {threading_text}")

	dives = []
	start_search_at_idx = 0

	while True:
		frame_gen = frame_generator(video_path)  # Use native framerate
		for _ in range(start_search_at_idx):
			try:
				next(frame_gen)
			except StopIteration:
				frame_gen = None # Generator is exhausted
				break

		if frame_gen is None:
			print("  -> No more dives found in the remainder of the video.")
			break

		print(f"Searching for a dive from frame {start_search_at_idx}...")
		start_idx, end_idx, confidence = find_next_dive(
			frame_gen, board_y_norm, water_y_norm,
			splash_zone_top_norm, splash_zone_bottom_norm, diver_zone_norm,
			start_search_at_idx=start_search_at_idx, debug=debug, splash_method=splash_method,
			video_fps=video_fps, use_threading=use_threading, enable_pose_optimization=enable_pose_optimization
		)

		if start_idx is not None and end_idx is not None:
			min_dive_frames = int(1.0 * video_fps)  # Dynamic minimum based on framerate
			if (end_idx - start_idx) >= min_dive_frames:
				confidence_text = f"({confidence} confidence)" if confidence == 'low' else ""
				print(f"  -> Found dive: Start {start_idx}, End {end_idx} {confidence_text}")
				dives.append((start_idx, end_idx, confidence))
			else:
				print(f"  -> Dive ignored (too short): Start {start_idx}, End {end_idx}")
			start_search_at_idx = end_idx + 1
		else:
			print("  -> No more dives found in the remainder of the video.")
			break
	return dives

def extract_and_save_dive(video_path, dive_number, start_idx, end_idx, confidence, output_dir, diver_zone_norm=None, video_fps=None, show_pose_overlay=False, preserve_audio=True):
	"""
	Extracts a single dive, optionally annotates it with pose, and saves it as a separate video file.
	Uses ROI processing for efficiency when diver_zone_norm is provided.
	Uses native video framerate if video_fps is not specified.
	Returns the actual extraction time for metrics tracking.

	Args:
		show_pose_overlay: If True, draws pose landmarks on the video
		preserve_audio: If True, uses FFmpeg to preserve audio from the source video
	"""
	extraction_start_time = time.time()

	if video_fps is None:
		video_fps = get_video_fps(video_path)

	confidence_suffix = "_low_conf" if confidence == 'low' else ""
	output_path = os.path.join(output_dir, f"dive_{dive_number+1}{confidence_suffix}.mp4")
	print(f"Extracting dive {dive_number+1} to {output_path} (frames {start_idx}-{end_idx}, {confidence} confidence) at {video_fps:.1f}fps")

	if show_pose_overlay:
		print(f"  → Pose overlay: enabled")
	else:
		print(f"  → Pose overlay: disabled")

	if preserve_audio:
		print(f"  → Audio preservation: enabled (requires FFmpeg)")
	else:
		print(f"  → Audio preservation: disabled")

	# Get the first frame to determine video dimensions
	try:
		_, first_frame = next(frame_generator(video_path))  # Use native framerate
		h, w = first_frame.shape[:2]
	except StopIteration:
		raise ValueError("Cannot extract from an empty video.")

	# Determine temporary output path for video processing
	temp_output_path = output_path
	if preserve_audio:
		temp_output_path = output_path.replace('.mp4', '_temp_video_only.mp4')

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(temp_output_path, fourcc, video_fps, (w, h))

	# Initialize pose detection only if needed
	mp_pose = None
	mp_drawing = None
	pose = None

	if show_pose_overlay:
		mp_pose = mp.solutions.pose
		mp_drawing = mp.solutions.drawing_utils
		# Suppress stderr completely during pose initialization to avoid repeated MediaPipe logs
		with suppress_stderr():
			pose = mp_pose.Pose(static_image_mode=True)

	# Create a fresh generator for processing
	dive_frame_gen = frame_generator(video_path)  # Use native framerate

	for idx, frame in dive_frame_gen:
		if idx < start_idx:
			continue
		if idx > end_idx:
			break

		# It's good practice to work on a copy
		annotated_frame = frame.copy()
		h, w = annotated_frame.shape[:2]

		# Draw pose landmarks - use ROI processing for efficiency (only if enabled)
		if show_pose_overlay and pose is not None:
			rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

			if diver_zone_norm is not None:
				# Extract diver detection zone coordinates
				left_norm, top_norm, right_norm, bottom_norm = diver_zone_norm
				left = int(left_norm * w)
				top = int(top_norm * h)
				right = int(right_norm * w)
				bottom = int(bottom_norm * h)

				# Extract ROI and process only that region
				roi = rgb[top:bottom, left:right]
				if roi.size > 0:
					res = pose.process(roi)
					if res.pose_landmarks:
						# Manually draw landmarks by converting ROI coordinates to full frame
						for landmark in res.pose_landmarks.landmark:
							# Convert from ROI coordinates to full frame coordinates
							full_x = int(landmark.x * (right - left) + left)
							full_y = int(landmark.y * (bottom - top) + top)
							cv2.circle(annotated_frame, (full_x, full_y), 3, (0,255,0), -1)

						# Draw connections manually for key body parts
						landmarks = res.pose_landmarks.landmark
						# Draw connections manually for key body parts
						landmarks = res.pose_landmarks.landmark
						connections = [
							# Body outline
							(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
							(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
							(mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
							(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
							(mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
							(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
							(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
							(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
							(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
							(mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
							(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
							(mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
						]

						for connection in connections:
							start_lm = landmarks[connection[0]]
							end_lm = landmarks[connection[1]]

							start_x = int(start_lm.x * (right - left) + left)
							start_y = int(start_lm.y * (bottom - top) + top)
							end_x = int(end_lm.x * (right - left) + left)
							end_y = int(end_lm.y * (bottom - top) + top)

							cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
			else:
				# Fallback: process entire frame
				res = pose.process(rgb)
				if res.pose_landmarks:
					mp_drawing.draw_landmarks(
						annotated_frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
						mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
						mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
					)

		# Add takeoff/entry text with confidence indicator
		if idx == start_idx:
			cv2.putText(annotated_frame, "DIVE START", (50,50),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
		elif idx == end_idx:
			entry_text = "DIVE END"
			if confidence == 'low':
				entry_text += " (LOW CONF)"
			color = (0,165,255) if confidence == 'low' else (0,0,255)  # Orange for low confidence
			cv2.putText(annotated_frame, entry_text, (50,100),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

		out.write(annotated_frame)

	# Clean up pose resources if used
	if pose is not None:
		pose.close()
	out.release()

	# Handle audio preservation using FFmpeg
	if preserve_audio:
		print(f"  🎵 Merging audio with video using FFmpeg...")
		try:
			# Calculate time range for audio extraction
			start_time_seconds = start_idx / video_fps
			duration_seconds = (end_idx - start_idx + 1) / video_fps

			# Use FFmpeg to extract the video segment with audio
			ffmpeg_cmd = [
				'ffmpeg', '-y',  # -y to overwrite output file
				'-i', video_path,  # Input video with audio
				'-i', temp_output_path,  # Input video-only file
				'-ss', str(start_time_seconds),  # Start time
				'-t', str(duration_seconds),  # Duration
				'-c:v', 'copy',  # Copy video from the video-only file
				'-c:a', 'aac',  # Encode audio to AAC
				'-map', '1:v:0',  # Use video from second input (our processed video)
				'-map', '0:a:0',  # Use audio from first input (original video)
				'-shortest',  # Stop when shortest stream ends
				output_path
			]

			# Run FFmpeg command
			result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)

			if result.returncode == 0:
				# Remove temporary video-only file
				os.remove(temp_output_path)
				print(f"  ✅ Audio successfully preserved in {output_path}")
			else:
				print(f"  ⚠️  FFmpeg failed, keeping video-only version:")
				print(f"     Error: {result.stderr}")
				# Rename temp file to final output
				os.rename(temp_output_path, output_path)

		except subprocess.TimeoutExpired:
			print(f"  ⚠️  FFmpeg timed out, keeping video-only version")
			os.rename(temp_output_path, output_path)
		except FileNotFoundError:
			print(f"  ⚠️  FFmpeg not found, keeping video-only version")
			print(f"     Install FFmpeg to enable audio preservation")
			os.rename(temp_output_path, output_path)
		except Exception as e:
			print(f"  ⚠️  Audio preservation failed: {e}")
			os.rename(temp_output_path, output_path)

	extraction_end_time = time.time()
	extraction_duration = extraction_end_time - extraction_start_time

	print(f"Successfully saved {output_path}")
	return extraction_duration

# ==================== USER INTERFACE FUNCTIONS ====================
def get_board_y_px(frame):
	"""
	Show the frame and let the user click to select the board y coordinate.
	Returns the y coordinate (int).
	Displays a moving horizontal line under the mouse.
	Optimized for better performance.
	"""
	clicked = {'y': None}
	mouse_y = {'y': None, 'changed': False}

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
		# Scale mouse coordinates back to original frame
		actual_y = int(y / display_scale) if display_scale < 1.0 else y
		if mouse_y['y'] != actual_y:
			mouse_y['y'] = actual_y
			mouse_y['changed'] = True

		if event == cv2.EVENT_LBUTTONDOWN:
			clicked['y'] = actual_y
			cv2.destroyWindow("Select Board Line")

	cv2.namedWindow("Select Board Line")
	cv2.setMouseCallback("Select Board Line", on_mouse)
	print("Click on the board line in the image window. Press 'a' to auto-detect.")

	last_draw = None
	while clicked['y'] is None:
		if mouse_y['changed'] or last_draw is None:
			display = clone.copy()
			if mouse_y['y'] is not None:
				# Scale y coordinate for display
				display_y = int(mouse_y['y'] * display_scale) if display_scale < 1.0 else mouse_y['y']
				cv2.line(display, (0, display_y), (display.shape[1], display_y), (0,255,255), 2)
			cv2.imshow("Select Board Line", display)
			mouse_y['changed'] = False
			last_draw = mouse_y['y']

		key = cv2.waitKey(50) & 0xFF  # Reduced frequency
		if key == 27:	# ESC to quit
			break
		elif key == ord('a'):
			board_y = auto_detect_board_y(frame)  # Use original frame for detection
			if board_y is not None:
				clicked['y'] = board_y
				print(f"Auto-detected board y: {board_y}")
				# Show detection result
				display_y = int(board_y * display_scale) if display_scale < 1.0 else board_y
				cv2.line(display, (0, display_y), (display.shape[1], display_y), (255,0,0), 2)
				cv2.imshow("Select Board Line", display)
				cv2.waitKey(700)
				break
	cv2.destroyAllWindows()
	return clicked['y']

def get_splash_zone(frame):
	"""
	Show the frame and let the user click and drag to select the splash detection zone.
	Returns (top_y, bottom_y) coordinates for the splash zone.
	Optimized for better performance.
	"""
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

	cv2.namedWindow("Select Splash Zone")
	cv2.setMouseCallback("Select Splash Zone", on_mouse)
	print("Click and drag to select the splash detection zone.")
	print("The zone should cover the area where water entry splashes occur (typically near the water surface).")

	last_state = None
	while clicked['top_y'] is None or clicked['bottom_y'] is None:
		# Only redraw if something changed
		current_state = (mouse_pos['y'], drawing['active'], drawing['start_y'])
		if mouse_pos['changed'] or last_state != current_state:
			display = clone.copy()

			# Show current mouse position line
			if mouse_pos['y'] is not None:
				display_y = int(mouse_pos['y'] * display_scale) if display_scale < 1.0 else mouse_pos['y']
				cv2.line(display, (0, display_y), (display.shape[1], display_y), (128,128,128), 1)

			# Show selection rectangle while dragging
			if drawing['active'] and drawing['start_y'] is not None and mouse_pos['y'] is not None:
				display_start_y = int(drawing['start_y'] * display_scale) if display_scale < 1.0 else drawing['start_y']
				display_mouse_y = int(mouse_pos['y'] * display_scale) if display_scale < 1.0 else mouse_pos['y']

				top = min(display_start_y, display_mouse_y)
				bottom = max(display_start_y, display_mouse_y)

				# Draw the splash zone rectangle similar to debug mode
				overlay = display.copy()
				cv2.rectangle(overlay, (0, top), (display.shape[1], bottom), (255,0,255), 2)
				cv2.rectangle(overlay, (0, top), (display.shape[1], bottom), (255,0,255), -1)
				cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

				# Calculate actual height in original frame coordinates
				actual_height = abs(drawing['start_y'] - mouse_pos['y'])
				cv2.putText(display, f"Splash Zone (height: {actual_height}px)", (20, top-10 if top > 30 else bottom+30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)

			cv2.putText(display, "Click and drag to select splash zone (ESC to skip)", (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
			cv2.putText(display, "Recommended: Select area around water surface", (10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
			cv2.imshow("Select Splash Zone", display)

			mouse_pos['changed'] = False
			last_state = current_state

		key = cv2.waitKey(50) & 0xFF  # Reduced frequency
		if key == 27:	# ESC to quit
			break

	cv2.destroyAllWindows()
	return clicked['top_y'], clicked['bottom_y']

def get_diver_detection_zone(frame):
	"""
	Show the frame and let the user click and drag to select the diver detection zone.
	Returns (left, top, right, bottom) coordinates for the bounding box.
	Optimized for better performance.
	"""
	clicked = {'coords': None}
	mouse_pos = {'x': None, 'y': None, 'changed': False}
	drawing = {'active': False, 'start_x': None, 'start_y': None}

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
			drawing['start_x'] = actual_x
			drawing['start_y'] = actual_y
		elif event == cv2.EVENT_LBUTTONUP and drawing['active']:
			drawing['active'] = False
			left = min(drawing['start_x'], actual_x)
			right = max(drawing['start_x'], actual_x)
			top = min(drawing['start_y'], actual_y)
			bottom = max(drawing['start_y'], actual_y)
			clicked['coords'] = (left, top, right, bottom)
			cv2.destroyWindow("Select Diver Detection Zone")

	cv2.namedWindow("Select Diver Detection Zone")
	cv2.setMouseCallback("Select Diver Detection Zone", on_mouse)
	print("Click and drag to select the diver detection zone.")
	print("This should include the diving board/platform where divers stand before diving.")
	print("Make sure to include the full area where a diver might be positioned.")

	last_state = None
	while clicked['coords'] is None:
		# Only redraw if something changed
		current_state = (mouse_pos['x'], mouse_pos['y'], drawing['active'], drawing['start_x'], drawing['start_y'])
		if mouse_pos['changed'] or last_state != current_state:
			display = clone.copy()

			# Show current mouse position crosshair
			if mouse_pos['x'] is not None and mouse_pos['y'] is not None:
				display_x = int(mouse_pos['x'] * display_scale) if display_scale < 1.0 else mouse_pos['x']
				display_y = int(mouse_pos['y'] * display_scale) if display_scale < 1.0 else mouse_pos['y']
				cv2.line(display, (display_x, 0), (display_x, display.shape[0]), (128,128,128), 1)
				cv2.line(display, (0, display_y), (display.shape[1], display_y), (128,128,128), 1)

			# Show selection rectangle while dragging
			if drawing['active'] and drawing['start_x'] is not None and drawing['start_y'] is not None and mouse_pos['x'] is not None and mouse_pos['y'] is not None:
				# Convert to display coordinates
				display_start_x = int(drawing['start_x'] * display_scale) if display_scale < 1.0 else drawing['start_x']
				display_start_y = int(drawing['start_y'] * display_scale) if display_scale < 1.0 else drawing['start_y']
				display_mouse_x = int(mouse_pos['x'] * display_scale) if display_scale < 1.0 else mouse_pos['x']
				display_mouse_y = int(mouse_pos['y'] * display_scale) if display_scale < 1.0 else mouse_pos['y']

				left = min(display_start_x, display_mouse_x)
				right = max(display_start_x, display_mouse_x)
				top = min(display_start_y, display_mouse_y)
				bottom = max(display_start_y, display_mouse_y)

				# Draw the detection zone rectangle
				cv2.rectangle(display, (left, top), (right, bottom), (0,255,0), 2)
				# Add semi-transparent overlay
				overlay = display.copy()
				cv2.rectangle(overlay, (left, top), (right, bottom), (0,255,0), -1)
				cv2.addWeighted(overlay, 0.1, display, 0.9, 0, display)

				# Show dimensions in original frame coordinates
				actual_width = abs(drawing['start_x'] - mouse_pos['x'])
				actual_height = abs(drawing['start_y'] - mouse_pos['y'])
				cv2.putText(display, f"Detection Zone ({actual_width}x{actual_height}px)",
					(left, top-10 if top > 30 else bottom+30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

			cv2.putText(display, "Click and drag to select diver detection zone (ESC to skip)", (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
			cv2.putText(display, "Include diving board/platform area", (10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
			cv2.imshow("Select Diver Detection Zone", display)

			mouse_pos['changed'] = False
			last_state = current_state

		key = cv2.waitKey(50) & 0xFF  # Reduced frequency
		if key == 27:	# ESC to quit
			break

	cv2.destroyAllWindows()
	return clicked['coords']

# ==================== MAIN APPLICATION ====================
def main():
	# Set up argument parser
	parser = argparse.ArgumentParser(description="Analyze a diving video to extract individual dives.")
	parser.add_argument("video_path", help="Path to the input video file.")
	parser.add_argument("--output_dir", default="data/pose_analysis/extracted_dives", help="Directory to save extracted dives.")
	parser.add_argument("--splash_method", default="motion_intensity",
						choices=['frame_diff', 'optical_flow', 'contour', 'motion_intensity', 'combined'],
						help="Splash detection method to use (default: motion_intensity)")
	parser.add_argument("--debug", action="store_true",
						help="Enable debug window for visual analysis (slower but helpful for tuning)")
	parser.add_argument("--no-threading", action="store_true",
						help="Disable threaded processing (use for debugging performance issues)")
	parser.add_argument("--show-pose-overlay", action="store_true",
						help="Enable pose overlay in extracted dive videos (disabled by default)")
	parser.add_argument("--no-audio", action="store_true",
						help="Disable audio preservation in extracted dive videos (enabled by default)")
	args = parser.parse_args()

	video_path = args.video_path
	output_dir = args.output_dir
	splash_method = args.splash_method
	debug = args.debug
	use_threading = not args.no_threading  # Default to True, disable with --no-threading
	show_pose_overlay = args.show_pose_overlay  # Default to False, enable with --show-pose-overlay
	preserve_audio = not args.no_audio  # Default to True, disable with --no-audio

	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)

	# Step 1: Get the first frame for user calibration
	print(f"Opening video file: {video_path}...")

	# Get video framerate first
	video_fps = get_video_fps(video_path)
	print(f"Detected video framerate: {video_fps:.1f}fps")

	try:
		first_frame_gen = frame_generator(video_path)  # Use native framerate
		_, first_frame = next(first_frame_gen)
		del first_frame_gen # free up the generator
	except StopIteration:
		print("Could not read the first frame. The video may be empty or corrupted.")
		return

	# Step 2: user specifies board_y_px on the first frame
	print("Please specify the diving board line.")
	board_y_px = get_board_y_px(first_frame)
	if board_y_px is None:
		print("No board line selected. Exiting.")
		return

	# Step 2.5: user specifies diver detection zone
	print("Please specify the diver detection zone.")
	diver_zone_coords = get_diver_detection_zone(first_frame)
	if diver_zone_coords is None:
		print("No diver detection zone selected. Using full frame detection.")
		diver_zone_norm = None
	else:
		left, top, right, bottom = diver_zone_coords
		h, w = first_frame.shape[:2]
		diver_zone_norm = (left/w, top/h, right/w, bottom/h)  # Normalize coordinates
		zone_width = right - left
		zone_height = bottom - top
		print(f"✓ Diver detection zone selected: ({left},{top}) to ({right},{bottom})")
		print(f"  Zone size: {zone_width}x{zone_height}px")
		print(f"  Normalized coordinates: ({diver_zone_norm[0]:.3f},{diver_zone_norm[1]:.3f}) to ({diver_zone_norm[2]:.3f},{diver_zone_norm[3]:.3f})")

	# Step 3: user specifies splash zone
	print("Please specify the splash detection zone.")
	splash_top_px, splash_bottom_px = get_splash_zone(first_frame)
	if splash_top_px is None or splash_bottom_px is None:
		print("No splash zone selected. Using default zone.")
		splash_zone_top_norm = None
		splash_zone_bottom_norm = None
	else:
		splash_zone_top_norm = splash_top_px / first_frame.shape[0]
		splash_zone_bottom_norm = splash_bottom_px / first_frame.shape[0]
		zone_height = splash_bottom_px - splash_top_px
		print(f"✓ Splash zone selected: {splash_top_px} to {splash_bottom_px} pixels (height: {zone_height}px)")
		print(f"  Normalized coordinates: {splash_zone_top_norm:.3f} to {splash_zone_bottom_norm:.3f}")

	# convert to normalized y
	board_y_norm = board_y_px / first_frame.shape[0]
	water_y_norm = 0.95  # bottom 5% is water

	# Step 4: Real-time dive detection and extraction
	print(f"\nStarting real-time dive detection and extraction using '{splash_method}' splash detection...")
	threading_status = "with threading enabled" if use_threading else "with threading disabled"
	print(f"🚀 Performance mode: {threading_status}")
	if debug:
		print("🐛 Debug mode enabled - showing visual analysis window")
		print("Legend: WAITING -> DIVER_ON_PLATFORM -> DIVING -> END")
	else:
		print("⚡ Fast mode - no debug window (use --debug for visual analysis)")
		print("Legend: WAITING -> DIVER_ON_PLATFORM -> DIVING -> END")

	print(f"💡 Real-time mode: Extraction will start immediately when each dive is detected!")

	dives = detect_and_extract_dives_realtime(
		video_path, board_y_norm, water_y_norm,
		splash_zone_top_norm=splash_zone_top_norm,
		splash_zone_bottom_norm=splash_zone_bottom_norm,
		diver_zone_norm=diver_zone_norm,
		debug=debug,
		splash_method=splash_method,
		use_threading=use_threading,
		output_dir=output_dir,
		show_pose_overlay=show_pose_overlay,
		preserve_audio=preserve_audio
	)

	# Handle both old and new return formats for compatibility
	if isinstance(dives, dict) and 'dives' in dives:
		result = dives
		dives_list = result['dives']
		metrics = result.get('metrics', {})
	else:
		# Backward compatibility - old format returned just the list
		dives_list = dives
		metrics = {}

	if not dives_list:
		print("\nNo dives were detected in the video.")
		return

	print(f"\n🎉 Real-time processing complete!")
	print(f"📊 Summary: {len(dives_list)} dives detected and extracted")
	print(f"📁 All dive videos saved in '{output_dir}'")
	print(f"💡 Real-time advantage: Extraction started immediately upon detection, maximizing parallel processing efficiency!")

	# Display detailed metrics if available
	if metrics:
		print("\nDetailed analysis metrics:")
		print_detailed_metrics(metrics, output_dir)


if __name__ == "__main__":
	main()
