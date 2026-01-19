"""
Person detection using YOLO-nano for dive validation.

Phase 3: Person presence tracking in video for third validation signal.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GPUInfo:
    """Information about available GPU device."""

    device_type: str  # 'cuda', 'metal', 'rocm', 'cpu'
    device_index: int  # GPU device index (0 if CPU)
    device_name: str  # Device name/description
    total_memory_mb: float  # Total GPU memory in MB
    available_memory_mb: float  # Available GPU memory in MB
    is_available: bool  # Whether device is accessible


def detect_gpu_device(force_cpu: bool = False) -> GPUInfo:
    """
    Detect available GPU device (CUDA/Metal/ROCm) with fallback to CPU.

    Args:
        force_cpu: Force CPU usage even if GPU is available

    Returns:
        GPUInfo object with device details
    """
    if force_cpu:
        return GPUInfo(
            device_type="cpu",
            device_index=0,
            device_name="CPU (forced)",
            total_memory_mb=0.0,
            available_memory_mb=0.0,
            is_available=True,
        )

    try:
        import torch

        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            device_index = 0  # Default to first GPU
            device_name = torch.cuda.get_device_name(device_index)
            total_memory = torch.cuda.get_device_properties(device_index).total_memory
            total_memory_mb = total_memory / (1024 * 1024)

            try:
                available_memory = torch.cuda.mem_get_info(device_index)[0]
                available_memory_mb = available_memory / (1024 * 1024)
            except Exception:
                available_memory_mb = total_memory_mb * 0.8  # Estimate

            return GPUInfo(
                device_type="cuda",
                device_index=device_index,
                device_name=f"NVIDIA {device_name}",
                total_memory_mb=total_memory_mb,
                available_memory_mb=available_memory_mb,
                is_available=True,
            )

        # Check for Metal (Apple Silicon)
        # Note: Metal support in PyTorch is via mps backend
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return GPUInfo(
                device_type="metal",
                device_index=0,
                device_name="Apple Metal (MPS)",
                total_memory_mb=0.0,  # Metal doesn't expose memory limits
                available_memory_mb=0.0,
                is_available=True,
            )

        # Check for ROCm (AMD)
        try:
            if torch.version.hip is not None:
                device_index = 0
                device_name = torch.cuda.get_device_name(device_index)
                total_memory = torch.cuda.get_device_properties(device_index).total_memory
                total_memory_mb = total_memory / (1024 * 1024)

                try:
                    available_memory = torch.cuda.mem_get_info(device_index)[0]
                    available_memory_mb = available_memory / (1024 * 1024)
                except Exception:
                    available_memory_mb = total_memory_mb * 0.8

                return GPUInfo(
                    device_type="rocm",
                    device_index=device_index,
                    device_name=f"AMD ROCm {device_name}",
                    total_memory_mb=total_memory_mb,
                    available_memory_mb=available_memory_mb,
                    is_available=True,
                )
        except Exception:
            pass

    except ImportError:
        pass

    # Fallback to CPU
    return GPUInfo(
        device_type="cpu",
        device_index=0,
        device_name="CPU (no GPU detected)",
        total_memory_mb=0.0,
        available_memory_mb=0.0,
        is_available=True,
    )


def check_gpu_memory(model_size_mb: float, required_buffer_mb: float = 100.0) -> bool:
    """
    Check if GPU has sufficient memory for model and inference.

    Args:
        model_size_mb: Model size in MB
        required_buffer_mb: Additional buffer needed in MB

    Returns:
        True if sufficient memory, False otherwise
    """
    gpu_info = detect_gpu_device()

    if gpu_info.device_type == "cpu":
        return True  # CPU memory checking not implemented

    if gpu_info.device_type == "metal":
        return True  # Metal doesn't expose memory limits

    required_mb = model_size_mb + required_buffer_mb
    if gpu_info.available_memory_mb < required_mb:
        print(
            f"⚠️  Insufficient GPU memory: {gpu_info.available_memory_mb:.0f}MB available, "
            f"{required_mb:.0f}MB required"
        )
        return False

    return True


def load_yolo_model(
    model_name: str = "yolov8n.pt",
    use_gpu: bool = False,
    force_cpu: bool = False,
    use_fp16: bool = False,
) -> Tuple[Any, GPUInfo]:
    """
    Load YOLO-nano model for person detection with GPU support.

    Args:
        model_name: Model to use (yolov8n = nano, fastest)
        use_gpu: Use GPU if available
        force_cpu: Force CPU usage
        use_fp16: Use FP16 half-precision (GPU only)

    Returns:
        Tuple of (YOLO model instance, GPUInfo)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

    # Suppress YOLO verbose output
    os.environ["YOLO_VERBOSE"] = "False"

    # Detect GPU
    gpu_info = detect_gpu_device(force_cpu=force_cpu)

    model = YOLO(model_name)

    # Configure device
    if use_gpu and gpu_info.device_type != "cpu":
        try:
            # Map device type to PyTorch device string
            if gpu_info.device_type == "cuda":
                device = f"cuda:{gpu_info.device_index}"
            elif gpu_info.device_type == "metal":
                device = "mps"
            elif gpu_info.device_type == "rocm":
                device = f"cuda:{gpu_info.device_index}"
            else:
                device = "cpu"

            model.to(device)

            # Apply FP16 if requested
            if use_fp16 and gpu_info.device_type in ("cuda", "rocm"):
                try:
                    import torch

                    model.model.half()
                    print(f"  FP16 enabled on {gpu_info.device_name}")
                except Exception as e:
                    print(f"  ⚠️  FP16 failed: {e}, using FP32")

            print(f"  GPU: {gpu_info.device_name}")
            print(f"  Memory: {gpu_info.available_memory_mb:.0f}MB available")

        except Exception as e:
            print(f"⚠️  Failed to load on GPU: {e}")
            print("  Falling back to CPU")
            model.to("cpu")
            gpu_info = detect_gpu_device(force_cpu=True)
    else:
        model.to("cpu")
        if force_cpu:
            print("  GPU: CPU (forced)")
        elif use_gpu:
            print("  GPU: CPU (no GPU available)")

    return model, gpu_info


def detect_person_frames(
    video_path: str,
    sample_fps: float = 5.0,
    confidence_threshold: float = 0.5,
    use_gpu: bool = False,
    force_cpu: bool = False,
    use_fp16: bool = False,
) -> List[Tuple[float, bool, float]]:
    """
    Detect frames where person is present in video.

    Args:
        video_path: Path to video file
        sample_fps: Frames per second to sample (lower = faster)
        confidence_threshold: YOLO confidence threshold (0.0-1.0)
        use_gpu: Use GPU for inference
        force_cpu: Force CPU usage (overrides use_gpu)
        use_fp16: Use FP16 half-precision (GPU only)

    Returns:
        List of (timestamp, person_present, confidence)
    """
    video_path = str(Path(video_path).resolve())

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load YOLO model with GPU support
    model, gpu_info = load_yolo_model(
        use_gpu=use_gpu,
        force_cpu=force_cpu,
        use_fp16=use_fp16,
    )

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
_gpu_info_cache = None


def get_cached_yolo_model(use_gpu: bool = False, force_cpu: bool = False) -> Tuple[Any, GPUInfo]:
    """
    Get or create cached YOLO model.

    Args:
        use_gpu: Use GPU if available
        force_cpu: Force CPU usage

    Returns:
        Tuple of (YOLO model instance, GPUInfo)
    """
    global _model_cache, _gpu_info_cache
    if _model_cache is None:
        _model_cache, _gpu_info_cache = load_yolo_model(use_gpu=use_gpu, force_cpu=force_cpu)
    return _model_cache, _gpu_info_cache


def clear_model_cache() -> None:
    """Clear cached YOLO model."""
    global _model_cache, _gpu_info_cache
    _model_cache = None
    _gpu_info_cache = None
