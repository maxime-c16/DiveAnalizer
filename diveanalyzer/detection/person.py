"""
Person detection using YOLO-nano for dive validation.

Phase 3: Person presence tracking in video for third validation signal.
"""

import os
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np


def load_yolo_model(model_name: str = "yolov8n.pt", use_gpu: bool = False):
    """
    Load YOLO-nano model for person detection.

    Args:
        model_name: Model to use (yolov8n = nano, fastest)
        use_gpu: Use GPU if available

    Returns:
        YOLO model instance
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

    # Suppress YOLO verbose output
    os.environ["YOLO_VERBOSE"] = "False"

    model = YOLO(model_name)

    # Configure device
    if use_gpu:
        try:
            model.to("cuda")
        except Exception:
            print("⚠️  GPU not available, falling back to CPU")
            model.to("cpu")
    else:
        model.to("cpu")

    return model


def detect_person_frames(
    video_path: str,
    sample_fps: float = 5.0,
    confidence_threshold: float = 0.5,
    use_gpu: bool = False,
) -> List[Tuple[float, bool, float]]:
    """
    Detect frames where person is present in video.

    Args:
        video_path: Path to video file
        sample_fps: Frames per second to sample (lower = faster)
        confidence_threshold: YOLO confidence threshold (0.0-1.0)
        use_gpu: Use GPU for inference

    Returns:
        List of (timestamp, person_present, confidence)
    """
    video_path = str(Path(video_path).resolve())

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load YOLO model
    model = load_yolo_model(use_gpu=use_gpu)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame step for desired sample rate
    frame_step = max(1, int(fps / sample_fps))

    results = []
    frame_count = 0
    sampled_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample at desired FPS
            if frame_count % frame_step == 0:
                # Get timestamp
                timestamp = frame_count / fps

                # Run YOLO inference
                yolo_results = model.predict(frame, verbose=False)

                # Check for person class (class 0 in COCO)
                person_detected = False
                max_confidence = 0.0

                for result in yolo_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            conf = float(box.conf)

                            # COCO class 0 = person
                            if class_id == 0 and conf > max_confidence:
                                max_confidence = conf
                                if conf > confidence_threshold:
                                    person_detected = True

                results.append((timestamp, person_detected, max_confidence))
                sampled_count += 1

            frame_count += 1

            # Progress indicator
            if sampled_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Person detection: {progress:.0f}% ({sampled_count} frames)")

    finally:
        cap.release()

    return results


def find_person_transitions(
    person_timeline: List[Tuple[float, bool, float]],
    transition_type: str = "disappear",
) -> List[float]:
    """
    Find transitions in person presence (appear/disappear).

    Args:
        person_timeline: List of (timestamp, present, confidence) from detect_person_frames
        transition_type: 'appear', 'disappear', or 'both'

    Returns:
        List of transition timestamps
    """
    transitions = []

    for i in range(1, len(person_timeline)):
        prev_time, prev_present, _ = person_timeline[i - 1]
        curr_time, curr_present, _ = person_timeline[i]

        # Check for transition
        if prev_present and not curr_present:  # Disappear
            if transition_type in ("disappear", "both"):
                transitions.append(curr_time)

        elif not prev_present and curr_present:  # Appear
            if transition_type in ("appear", "both"):
                transitions.append(curr_time)

    return transitions


def smooth_person_timeline(
    person_timeline: List[Tuple[float, bool, float]],
    window_size: int = 2,
) -> List[Tuple[float, bool, float]]:
    """
    Smooth person presence timeline to remove jitter.

    Args:
        person_timeline: Raw detection timeline
        window_size: Smoothing window size (frames)

    Returns:
        Smoothed timeline
    """
    if len(person_timeline) < window_size:
        return person_timeline

    smoothed = []

    for i in range(len(person_timeline)):
        # Get window around current frame
        start = max(0, i - window_size // 2)
        end = min(len(person_timeline), i + window_size // 2 + 1)
        window = person_timeline[start:end]

        # Vote on presence (majority wins)
        present_count = sum(1 for _, p, _ in window if p)
        avg_present = present_count > len(window) // 2

        # Average confidence in window
        avg_confidence = np.mean([c for _, _, c in window])

        # Use original timestamp
        timestamp = person_timeline[i][0]
        smoothed.append((timestamp, avg_present, avg_confidence))

    return smoothed


def find_person_zone_departures(
    person_timeline: List[Tuple[float, bool, float]],
    min_absence_duration: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Find moments when person leaves diving zone (disappears from frame).

    These are potential dive starts.

    Args:
        person_timeline: Smoothed person presence timeline
        min_absence_duration: Minimum time person must be absent to count as departure

    Returns:
        List of (departure_time, confidence)
    """
    departures = []
    in_absence = False
    absence_start = 0.0
    absence_confidence = 0.0

    for timestamp, present, confidence in person_timeline:
        if not present and not in_absence:
            # Start of absence
            in_absence = True
            absence_start = timestamp
            absence_confidence = confidence

        elif present and in_absence:
            # End of absence
            absence_duration = timestamp - absence_start
            if absence_duration >= min_absence_duration:
                # Valid departure
                departures.append((absence_start, absence_confidence))
            in_absence = False

    # If video ends during absence, count it
    if in_absence and (person_timeline[-1][0] - absence_start) >= min_absence_duration:
        departures.append((absence_start, absence_confidence))

    return departures


def estimate_person_confidence(
    person_timeline: List[Tuple[float, bool, float]]
) -> float:
    """
    Estimate overall person detection confidence from timeline.

    Returns:
        Confidence score (0.0-1.0)
    """
    if not person_timeline:
        return 0.0

    # Average confidence when person is detected
    detected_confidences = [c for _, present, c in person_timeline if present]

    if not detected_confidences:
        return 0.0

    return float(np.mean(detected_confidences))


def get_person_presence_percentage(
    person_timeline: List[Tuple[float, bool, float]]
) -> float:
    """
    Calculate percentage of time person is visible.

    Returns:
        Percentage (0-100)
    """
    if not person_timeline:
        return 0.0

    present_count = sum(1 for _, present, _ in person_timeline if present)
    total = len(person_timeline)

    return (present_count / total) * 100 if total > 0 else 0.0


# Module-level model cache
_model_cache = None


def get_cached_yolo_model(use_gpu: bool = False):
    """Get or create cached YOLO model."""
    global _model_cache
    if _model_cache is None:
        _model_cache = load_yolo_model(use_gpu=use_gpu)
    return _model_cache


def clear_model_cache():
    """Clear cached YOLO model."""
    global _model_cache
    _model_cache = None
