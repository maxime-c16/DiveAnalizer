import os
import sys

# Comprehensive log suppression for MediaPipe and TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # Suppress all TensorFlow logs
os.environ['GLOG_minloglevel'] = '3'         # Suppress Google logs (MediaPipe)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'    # Disable oneDNN optimization messages
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'    # Disable GPU to prevent GL context logs

# Create a custom stderr filter to suppress specific MediaPipe/TensorFlow messages
class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filtered_phrases = [
            'GL version:',
            'INFO: Created TensorFlow',
            'WARNING: All log messages before absl',
            'gl_context.cc:',
            'TensorFlow Lite XNNPACK delegate',
            'I0000',
            'W0000',
            'renderer:',
            'Intel(R)',
            'INTEL-',
            'OpenGL',
            'Graphics'
        ]

    def write(self, text):
        # Only write if the text doesn't contain filtered phrases
        if not any(phrase in text for phrase in self.filtered_phrases):
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

    def fileno(self):
        return self.original_stderr.fileno()

# Install the stderr filter
sys.stderr = FilteredStderr(sys.stderr)

import cv2
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Temporarily suppress all stderr during MediaPipe import to avoid C++ level logging
import contextlib
import tempfile

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to completely suppress stderr at the file descriptor level"""
    # Save the original stderr file descriptor
    original_stderr_fd = sys.stderr.fileno()

    # Open a null device
    with open(os.devnull, 'w') as devnull:
        # Duplicate the original stderr file descriptor
        stderr_copy = os.dup(original_stderr_fd)

        try:
            # Redirect stderr to null device at file descriptor level
            os.dup2(devnull.fileno(), original_stderr_fd)
            yield
        finally:
            # Restore the original stderr
            os.dup2(stderr_copy, original_stderr_fd)
            os.close(stderr_copy)

# Import MediaPipe with stderr suppressed
with suppress_stderr():
    import mediapipe as mp
    mp_pose = mp.solutions.pose

mp_pose = mp.solutions.pose

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

	# print(f"Video framerate: {video_fps:.1f}fps, Processing at: {target_fps:.1f}fps (interval: {interval})")

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
			x1, y1, x2, y2 = l[0]
			# Only consider (nearly) horizontal lines away from the bottom
			if abs(y2 - y1) < 10 and min(y1,y2)>h//6 and max(y1,y2)<int(h*0.97):
				y_lines.append((y1 + y2) // 2)
		if y_lines:
			# The lowest horizontal except for the bottom of the image
			return int(sorted(y_lines)[-1])
	return None

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
	max_intensity = np.max(gradient_magnitude)
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
def find_next_dive(frame_gen, board_y_norm, water_y_norm=0.95, takeoff_thresh=0.9,
				   splash_zone_top_norm=None, splash_zone_bottom_norm=None,
				   diver_zone_norm=None, start_search_at_idx=0, debug=False, splash_method='combined',
				   video_fps=30):


	"""
	New dive detection logic:
	1. Start: When diver is detected inside the diver detection zone
	2. End: When splash detected (after min frames) AND diver no longer in zone
	3. Fallback: Auto-split after timeout if no splash but diver disappeared

	splash_method options: 'frame_diff', 'optical_flow', 'contour', 'motion_intensity', 'combined'
	Returns (start_idx, end_idx, confidence) where confidence is 'high' or 'low'
	"""

	# Suppress stderr during pose initialization to avoid repeated MediaPipe logs
	with suppress_stderr():
		pose = mp_pose.Pose(static_image_mode=True)

	# Dive states
	WAITING = 0
	DIVER_ON_PLATFORM = 1
	DIVING = 2

	state = WAITING
	start_idx = None
	end_idx = None
	confidence = 'high'

	# Dynamic timing constraints based on video framerate
	min_dive_frames = int(1.0 * video_fps)  # 1 second worth of frames
	max_no_splash_frames = int(6.0 * video_fps)  # 6 seconds worth of frames
	frames_since_start = 0
	frames_without_diver = 0

	print(f"Using dynamic timing: min_dive={min_dive_frames} frames, max_timeout={max_no_splash_frames} frames at {video_fps:.1f}fps")

	# Debug window management
	skipping_to_action = debug  # Start skipping if debug is enabled
	debug_frame_delay = max(1, int(1000 / video_fps))  # Frame delay for debug playback

	# For splash detection
	prev_gray = None
	splash_detected_idx = None
	splash_thresh = 12.0

	# Peak-based splash event detection for motion_intensity
	splash_event_state = 'waiting'  # 'waiting', 'in_splash', 'post_splash'
	splash_start_idx = None
	peak_score = 0
	frames_since_peak = 0
	cooldown_frames = 5  # Minimum frames between splash events

	# Diver detection in zone
	consecutive_diver_frames = 0
	consecutive_no_diver_frames = 0
	diver_detection_threshold = 3  # Need 3 consecutive frames to confirm diver presence/absence

	for idx, frame in frame_gen:
		h, w = frame.shape[:2]
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# --- Splash detection setup ---
		if splash_zone_top_norm is not None and splash_zone_bottom_norm is not None:
			band_top = int(splash_zone_top_norm * h)
			band_bot = int(splash_zone_bottom_norm * h)
		else:
			# Fallback to old behavior
			water_y_px = int(water_y_norm * h)
			splash_band_height = 70
			band_bot = water_y_px
			band_top = max(water_y_px - splash_band_height, 0)

		# Calculate splash score using selected method
		splash_band = gray[band_top:band_bot, :]
		splash_score = 0
		splash_this_frame = False
		splash_details = {}

		if prev_gray is not None:
			prev_band = prev_gray[band_top:band_bot, :]

			if splash_method == 'frame_diff':
				splash_this_frame, splash_score = detect_splash_frame_diff(splash_band, prev_band, splash_thresh)
			elif splash_method == 'optical_flow':
				splash_this_frame, splash_score = detect_splash_optical_flow(splash_band, prev_band)
			elif splash_method == 'contour':
				splash_this_frame, splash_score = detect_splash_contour_analysis(splash_band, prev_band)
			elif splash_method == 'motion_intensity':
				# Get raw score from motion_intensity method
				_, splash_score = detect_splash_motion_intensity(splash_band, prev_band)

				# Implement peak-based splash event detection
				splash_this_frame = False

				if splash_event_state == 'waiting':
					if splash_score > splash_thresh:
						# Start of new splash event
						splash_event_state = 'in_splash'
						splash_start_idx = idx
						peak_score = splash_score
						if debug:
							print(f"    Motion Intensity: Splash event started at frame {idx}, score: {splash_score:.1f}")

				elif splash_event_state == 'in_splash':
					if splash_score > peak_score:
						# Update peak
						peak_score = splash_score
						frames_since_peak = 0
					else:
						frames_since_peak += 1

					if splash_score <= splash_thresh:
						# End of splash event - mark this as a detection
						splash_event_state = 'post_splash'
						splash_this_frame = True  # This frame marks the end of the splash event
						if debug:
							print(f"    Motion Intensity: Splash event ended at frame {idx}, peak: {peak_score:.1f}")
						frames_since_peak = 0

				elif splash_event_state == 'post_splash':
					frames_since_peak += 1
					if frames_since_peak >= cooldown_frames:
						splash_event_state = 'waiting'
						if debug:
							print(f"    Motion Intensity: Ready for next splash detection after cooldown")

			elif splash_method == 'combined':
				splash_this_frame, splash_score, splash_details = detect_splash_combined(splash_band, prev_band, splash_thresh)

			if splash_this_frame:
				splash_detected_idx = idx
		prev_gray = gray

		# --- Diver detection in zone ---
		diver_in_zone = False
		pose_landmarks_full_frame = None

		if diver_zone_norm is not None:
			# Extract diver detection zone coordinates
			left_norm, top_norm, right_norm, bottom_norm = diver_zone_norm
			left = int(left_norm * w)
			top = int(top_norm * h)
			right = int(right_norm * w)
			bottom = int(bottom_norm * h)

			# Extract region of interest for pose detection - ONLY process the ROI
			roi = rgb[top:bottom, left:right]
			if roi.size > 0:
				res = pose.process(roi)
				if res.pose_landmarks:
					# Diver detected in the zone
					diver_in_zone = True
					# Convert ROI landmarks back to full frame coordinates for later use
					pose_landmarks_full_frame = []
					for landmark in res.pose_landmarks.landmark:
						# Convert from ROI coordinates to full frame coordinates
						full_x = (landmark.x * (right - left) + left) / w
						full_y = (landmark.y * (bottom - top) + top) / h
						pose_landmarks_full_frame.append((full_x, full_y, landmark.z))
		else:
			# Fallback: detect pose in entire frame (old behavior)
			res = pose.process(rgb)
			diver_in_zone = res.pose_landmarks is not None
			if res.pose_landmarks:
				pose_landmarks_full_frame = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]

		# Update diver presence counters
		if diver_in_zone:
			consecutive_diver_frames += 1
			consecutive_no_diver_frames = 0
		else:
			consecutive_diver_frames = 0
			consecutive_no_diver_frames += 1

		# --- State Machine Logic ---
		if state == WAITING:
			# Look for diver on platform
			if consecutive_diver_frames >= diver_detection_threshold:
				state = DIVER_ON_PLATFORM
				start_idx = idx - consecutive_diver_frames + 1  # Start from first detection
				frames_since_start = consecutive_diver_frames - 1
				if debug:
					print(f"Frame {start_idx}: Diver detected on platform - DIVE START")

		elif state == DIVER_ON_PLATFORM:
			frames_since_start += 1

			# Check if diver left the platform
			if consecutive_no_diver_frames >= diver_detection_threshold:
				state = DIVING
				frames_without_diver = consecutive_no_diver_frames
				if debug:
					print(f"Frame {idx}: Diver left platform - DIVING")

		elif state == DIVING:
			frames_since_start += 1

			if not diver_in_zone:
				frames_without_diver += 1
			else:
				frames_without_diver = 0  # Reset if diver reappears

			# Check for end conditions
			dive_long_enough = frames_since_start >= min_dive_frames

			if dive_long_enough:
				# High confidence end: Splash detected and diver not in zone
				if splash_this_frame or (splash_detected_idx is not None and abs(idx - splash_detected_idx) <= 3):
					if frames_without_diver >= diver_detection_threshold:
						end_idx = idx
						confidence = 'high'
						if debug:
							print(f"Frame {idx}: High confidence dive end - SPLASH + NO DIVER")
						break

				# Low confidence end: Timeout without splash but diver gone
				elif frames_without_diver >= max_no_splash_frames:
					end_idx = idx
					confidence = 'low'
					if debug:
						print(f"Frame {idx}: Low confidence dive end - TIMEOUT")
					break

		# --- Debug visualization ---
		if debug:
			# Only show debug window when we're in an interesting state or when diver is first detected
			show_debug = False

			if state == DIVER_ON_PLATFORM or state == DIVING:
				show_debug = True
			elif state == WAITING and consecutive_diver_frames >= diver_detection_threshold:
				# About to transition to DIVER_ON_PLATFORM
				show_debug = True

			# Print skipping message only once
			if skipping_to_action and not show_debug:
				if idx % 100 == 0:  # Print progress every 100 frames
					print(f"🔍 Debug mode: Skipping to next interesting moment (frame {idx})...")
			elif show_debug and skipping_to_action:
				skipping_to_action = False
				print(f"🎯 Debug window starting at frame {idx} - diver detected!")

			if show_debug:
				display_frame = frame.copy()

				# Draw splash zone
				cv2.rectangle(display_frame, (0, band_top), (w, band_bot), (255,0,255), 2)
				if splash_this_frame:
					overlay = display_frame.copy()
					cv2.rectangle(overlay, (0, band_top), (w, band_bot), (0,0,255), -1)
					cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

				# Draw diver detection zone
				if diver_zone_norm is not None:
					left_norm, top_norm, right_norm, bottom_norm = diver_zone_norm
					left = int(left_norm * w)
					top = int(top_norm * h)
					right = int(right_norm * w)
					bottom = int(bottom_norm * h)
					color = (0,255,0) if diver_in_zone else (0,255,255)
					cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

				# Status text
				status_text = f"State: {['WAITING', 'ON_PLATFORM', 'DIVING'][state]}"
				cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

				if state != WAITING:
					cv2.putText(display_frame, f"Frames since start: {frames_since_start}", (10, 60),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

				if state == DIVING:
					cv2.putText(display_frame, f"Frames without diver: {frames_without_diver}", (10, 90),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

				cv2.putText(display_frame, f"Diver in zone: {diver_in_zone}", (10, 120),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if diver_in_zone else (0,0,255), 2)

				cv2.putText(display_frame, f"Splash score: {splash_score:.1f}", (10, 150),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

				# Add frame number for reference
				cv2.putText(display_frame, f"Frame: {idx}", (10, 180),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

				# Add playback speed controls in debug
				cv2.putText(display_frame, "Debug Controls: SPACE=pause, +/-=speed, ESC=quit", (10, display_frame.shape[0]-20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

				cv2.imshow("Debug Dive Detection", display_frame)

				# Use the debug frame delay for smooth playback
				key = cv2.waitKey(debug_frame_delay) & 0xFF

				# Debug controls
				if key == 27:  # ESC to quit debug
					debug = False
					cv2.destroyWindow("Debug Dive Detection")
				elif key == ord(' '):  # SPACE to pause
					print("⏸️  Debug paused - press any key to continue...")
					cv2.waitKey(0)
				elif key == ord('+') or key == ord('='):  # Speed up (reduce delay)
					debug_frame_delay = max(1, debug_frame_delay // 2)
					print(f"⏩ Debug speed increased (delay: {debug_frame_delay}ms)")
				elif key == ord('-'):  # Slow down (increase delay)
					debug_frame_delay = min(1000, debug_frame_delay * 2)
					print(f"⏪ Debug speed decreased (delay: {debug_frame_delay}ms)")

	pose.close()
	if debug:
		cv2.destroyAllWindows()

	return start_idx, end_idx, confidence

def detect_all_dives(video_path, board_y_norm, water_y_norm, takeoff_thresh,
					 splash_zone_top_norm=None, splash_zone_bottom_norm=None,
					 diver_zone_norm=None, debug=False, splash_method='motion_intensity'):
	"""
	Detects all dive sequences in the video using the new detection logic.
	Returns a list of (start_idx, end_idx, confidence) tuples for each dive.
	"""

	# Get video framerate for dynamic timing
	video_fps = get_video_fps(video_path)
	print(f"Analyzing video at native framerate: {video_fps:.1f}fps")

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
			frame_gen, board_y_norm, water_y_norm, takeoff_thresh,
			splash_zone_top_norm, splash_zone_bottom_norm, diver_zone_norm,
			start_search_at_idx=start_search_at_idx, debug=debug, splash_method=splash_method,
			video_fps=video_fps
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

def extract_and_save_dive(video_path, dive_number, start_idx, end_idx, confidence, output_dir, diver_zone_norm=None, video_fps=None):
	"""
	Extracts a single dive, annotates it with pose, and saves it as a separate video file.
	Uses ROI processing for efficiency when diver_zone_norm is provided.
	Uses native video framerate if video_fps is not specified.
	"""
	if video_fps is None:
		video_fps = get_video_fps(video_path)

	confidence_suffix = "_low_conf" if confidence == 'low' else ""
	output_path = os.path.join(output_dir, f"dive_{dive_number+1}{confidence_suffix}.mp4")
	print(f"Extracting dive {dive_number+1} to {output_path} (frames {start_idx}-{end_idx}, {confidence} confidence) at {video_fps:.1f}fps")

	# Get the first frame to determine video dimensions
	try:
		_, first_frame = next(frame_generator(video_path))  # Use native framerate
		h, w = first_frame.shape[:2]
	except StopIteration:
		raise ValueError("Cannot extract from an empty video.")

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_path, fourcc, video_fps, (w, h))

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

		# Draw pose landmarks - use ROI processing for efficiency
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

	pose.close()
	out.release()
	print(f"Successfully saved {output_path}")


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

	def on_mouse(event, x, y, flags, param):
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

	def on_mouse(event, x, y, flags, param):
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

	def on_mouse(event, x, y, flags, param):
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

def calibrate_takeoff_thresh(video_path, board_y_norm, initial_thresh=0.1, water_y_norm=0.95, max_frames=100):
	"""
	Interactive tool to calibrate takeoff_thresh.
	Use UP/DOWN arrows to adjust threshold, 'q' to finish.
	"""
	# Suppress stderr during pose initialization to avoid repeated MediaPipe logs
	with suppress_stderr():
		pose = mp_pose.Pose(static_image_mode=True)
	thresh = initial_thresh
	frame_gen = frame_generator(video_path)

	# Store a limited number of frames for calibration
	calib_frames = []
	for i, (_, frame) in enumerate(frame_gen):
		if i >= max_frames:
			break
		calib_frames.append(frame)

	if not calib_frames:
		print("Warning: No frames to calibrate with.")
		return initial_thresh

	idx = 0
	while 0 <= idx < len(calib_frames):
		frame = calib_frames[idx]
		h, w = frame.shape[:2]
		board_y_px = int(board_y_norm * h)
		water_y_px = int(water_y_norm * h)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		res = pose.process(rgb)
		ankle_l, ankle_r = None, None
		if res.pose_landmarks:
			lm = res.pose_landmarks.landmark
			ankle_l = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y
			ankle_r = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y
		# Overlay
		disp = frame.copy()
		cv2.line(disp, (0, board_y_px), (w, board_y_px), (0,255,255), 2)
		cv2.line(disp, (0, water_y_px), (w, water_y_px), (255,128,0), 2)
		cv2.putText(disp, f"takeoff_thresh: {thresh:.3f} (UP/DOWN to adjust, q to quit)", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
		if ankle_l is not None and ankle_r is not None:
			ankle_l_px = int(ankle_l * h)
			ankle_r_px = int(ankle_r * h)
			cv2.circle(disp, (w//2-40, ankle_l_px), 8, (0,255,0), -1)
			cv2.circle(disp, (w//2+40, ankle_r_px), 8, (0,255,0), -1)
			# Show if takeoff would be detected
			if ankle_l < board_y_norm - thresh or ankle_r < board_y_norm - thresh:
				cv2.putText(disp, "TAKEOFF DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 3)
		cv2.imshow("Calibrate Takeoff Threshold", disp)
		key = cv2.waitKey(0)
		if key == ord('q'):
			break
		elif key == 82:  # UP arrow
			thresh += 0.01
		elif key == 84:  # DOWN arrow
			thresh = max(0.01, thresh - 0.01)
		elif key == 83:  # RIGHT arrow
			idx = min(idx+1, len(calib_frames)-1)
		elif key == 81:  # LEFT arrow
			idx = max(idx-1, 0)
		else:
			idx += 1
	cv2.destroyWindow("Calibrate Takeoff Threshold")
	return thresh

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
	args = parser.parse_args()

	video_path = args.video_path
	output_dir = args.output_dir
	splash_method = args.splash_method
	debug = args.debug

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

	# Step 4: detect all dives
	print(f"\nStarting dive detection with new algorithm using '{splash_method}' splash detection...")
	if debug:
		print("🐛 Debug mode enabled - showing visual analysis window")
		print("Legend: WAITING -> DIVER_ON_PLATFORM -> DIVING -> END")
	else:
		print("⚡ Fast mode - no debug window (use --debug for visual analysis)")
		print("Legend: WAITING -> DIVER_ON_PLATFORM -> DIVING -> END")
	dives = detect_all_dives(
		video_path, board_y_norm, water_y_norm, takeoff_thresh=0.1,
		splash_zone_top_norm=splash_zone_top_norm,
		splash_zone_bottom_norm=splash_zone_bottom_norm,
		diver_zone_norm=diver_zone_norm,
		debug=debug,
		splash_method=splash_method
	)

	if not dives:
		print("\nNo dives were detected in the video.")
		return

	print(f"\nDetected {len(dives)} dives. Now extracting and saving each one using parallel processing...")

	# Step 5: extract, annotate, and save each dive using threading
	import time
	start_time = time.time()

	# Determine optimal number of workers (max 4 to avoid overwhelming the system)
	max_workers = min(4, len(dives), os.cpu_count() or 2)
	print(f"Using {max_workers} worker threads for parallel extraction...")

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		# Submit all extraction tasks
		future_to_dive = {
			executor.submit(
				extract_and_save_dive,
				video_path, i, start_idx, end_idx, confidence,
				output_dir, diver_zone_norm, video_fps=video_fps
			): (i+1, start_idx, end_idx, confidence)
			for i, (start_idx, end_idx, confidence) in enumerate(dives)
		}

		# Track completion
		completed = 0
		total = len(dives)

		# Collect results as they complete
		for future in as_completed(future_to_dive):
			dive_num, start_idx, end_idx, confidence = future_to_dive[future]
			try:
				future.result()  # This will raise any exception that occurred
				completed += 1
				print(f"✅ Completed dive {dive_num}/{total} (frames {start_idx}-{end_idx}, {confidence} confidence) - {completed}/{total} done")
			except Exception as e:
				print(f"❌ Error extracting dive {dive_num}: {e}")
				completed += 1

	extraction_time = time.time() - start_time
	print(f"\n🎉 Extraction complete in {extraction_time:.1f}s! Dives saved in '{output_dir}'.")
	print(f"⚡ Parallel processing saved approximately {extraction_time * (len(dives) - 1) / len(dives):.1f}s compared to sequential processing.")


if __name__ == "__main__":
	main()
