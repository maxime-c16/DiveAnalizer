"""Motion detection module for DiveAnalyzer Phase 2.

Detects motion bursts by analyzing frame differences at low FPS on 480p proxy.
Used as secondary validation signal to reduce audio-only false positives.

Algorithm:
1. Load video at 5 FPS (low computational cost)
2. Compute grayscale frame difference
3. Find bursts of high-motion activity
4. Return timestamps and intensity of motion events
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


def detect_motion_bursts(
    video_path: str,
    sample_fps: float = 5.0,
    zone: Optional[Tuple[float, float, float, float]] = None,
    threshold_percentile: float = 80,
    min_burst_duration: float = 0.5,
    verbose: bool = False,
) -> List[Tuple[float, float, float]]:
    """Detect bursts of motion activity in video.

    Args:
        video_path: Path to video (use proxy for speed)
        sample_fps: Frames per second to sample (5 = analyze 1 frame every 200ms)
        zone: Optional (x1, y1, x2, y2) normalized coordinates for analysis area
              Example: (0.2, 0.3, 0.8, 0.9) for middle 60% of frame
        threshold_percentile: Motion above this percentile triggers burst
        min_burst_duration: Minimum duration of burst to report (seconds)
        verbose: Print progress

    Returns:
        List of (start_time, end_time, intensity) tuples for each motion burst

    Raises:
        ImportError: If cv2 (OpenCV) is not available
        RuntimeError: If video reading fails

    Example:
        >>> bursts = detect_motion_bursts('proxy_480p.mp4', sample_fps=5)
        >>> for start, end, intensity in bursts:
        ...     print(f"Motion burst: {start:.1f}s - {end:.1f}s (intensity: {intensity:.1f})")
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for motion detection. "
            "Install with: pip install opencv-python"
        )

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0

        frame_skip = max(1, int(fps / sample_fps))

        if verbose:
            print(f"Analyzing motion: {Path(video_path).name}")
            print(f"  Video: {total_frames} frames @ {fps:.1f} FPS ({video_duration:.1f}s)")
            print(f"  Sampling: 1 frame every {frame_skip} frames (~{sample_fps:.1f} FPS)")

    except Exception as e:
        raise RuntimeError(f"Failed to read video: {e}") from e

    # Extract motion scores
    motion_scores = []
    prev_gray = None
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            # Apply zone if specified (normalized coordinates)
            if zone is not None:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = zone
                frame = frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Compute frame difference
                diff = cv2.absdiff(gray, prev_gray)
                score = np.mean(diff)

                timestamp = frame_index / fps
                motion_scores.append((timestamp, score))

            prev_gray = gray
            frame_index += 1

    finally:
        cap.release()

    if not motion_scores:
        return []

    # Find motion bursts
    return _find_motion_bursts(
        motion_scores,
        threshold_percentile=threshold_percentile,
        min_burst_duration=min_burst_duration,
    )


def _find_motion_bursts(
    scores: List[Tuple[float, float]],
    threshold_percentile: float = 80,
    min_burst_duration: float = 0.5,
) -> List[Tuple[float, float, float]]:
    """Group consecutive high-motion frames into bursts.

    Args:
        scores: List of (timestamp, motion_score) tuples
        threshold_percentile: Percentile above which to trigger burst
        min_burst_duration: Minimum burst duration in seconds

    Returns:
        List of (start_time, end_time, max_intensity) tuples
    """
    if not scores:
        return []

    timestamps, values = zip(*scores)
    threshold = np.percentile(values, threshold_percentile)

    bursts = []
    burst_start = None
    burst_max = 0
    sample_interval = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1.0

    for t, v in scores:
        if v > threshold:
            if burst_start is None:
                burst_start = t
            burst_max = max(burst_max, v)
        else:
            if burst_start is not None:
                burst_duration = t - burst_start
                if burst_duration >= min_burst_duration:
                    bursts.append((burst_start, t, burst_max))
                burst_start = None
                burst_max = 0

    # Handle burst at end of video
    if burst_start is not None:
        last_t = timestamps[-1]
        burst_duration = last_t - burst_start
        if burst_duration >= min_burst_duration:
            bursts.append((burst_start, last_t, burst_max))

    return bursts


def find_motion_before_event(
    motion_events: List[Tuple[float, float, float]],
    event_time: float,
    min_window: float = 2.0,
    max_window: float = 12.0,
) -> Optional[Tuple[float, float, float]]:
    """Find motion activity before an audio peak (for fusion).

    Args:
        motion_events: List of (start, end, intensity) from detect_motion_bursts()
        event_time: Time of audio peak/event in seconds
        min_window: Minimum time before event to look (seconds)
        max_window: Maximum time before event to look (seconds)

    Returns:
        Motion event (start, end, intensity) if found in window, or None

    Example:
        >>> bursts = detect_motion_bursts('proxy.mp4')
        >>> motion = find_motion_before_event(bursts, event_time=4.5, min_window=2, max_window=12)
    """
    for start, end, intensity in motion_events:
        # Check if motion ends within the time window before event
        time_before = event_time - end
        if min_window <= time_before <= max_window:
            return (start, end, intensity)

    return None


def estimate_motion_zones(
    video_path: str,
    sample_frames: int = 30,
) -> dict:
    """Estimate zones of activity for interactive selection.

    Analyzes sample frames to identify active areas, helpful for calibration.

    Args:
        video_path: Path to video
        sample_frames: Number of sample frames to analyze

    Returns:
        Dictionary with zone recommendations and statistics
    """
    try:
        import cv2
    except ImportError:
        return {"error": "OpenCV required"}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, total_frames // sample_frames)

        # Collect motion data
        motion_map = None
        frame_index = 0
        sampled = 0

        while sampled < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if motion_map is None:
                motion_map = np.zeros_like(gray, dtype=np.float32)

            # Accumulate motion information
            motion_map += (gray.astype(np.float32) / 255.0)
            sampled += 1
            frame_index += 1

        cap.release()

        if motion_map is None:
            return {"error": "Could not extract frames"}

        # Find active regions
        motion_map /= sampled
        h, w = motion_map.shape

        # Divide into 3x3 grid and find which regions are active
        grid_height = h // 3
        grid_width = w // 3

        zones = {}
        for gy in range(3):
            for gx in range(3):
                region = motion_map[
                    gy*grid_height:(gy+1)*grid_height,
                    gx*grid_width:(gx+1)*grid_width
                ]
                activity = float(np.mean(region))
                zones[f"zone_{gy}_{gx}"] = {
                    "position": [gx/3, gy/3, (gx+1)/3, (gy+1)/3],
                    "activity_level": activity,
                }

        return {
            "frame_resolution": (w, h),
            "zones": zones,
            "recommended_zone": [0.0, 0.0, 1.0, 1.0],  # Full frame default
        }

    except Exception as e:
        return {"error": str(e)}
