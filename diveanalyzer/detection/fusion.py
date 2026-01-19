"""
Signal fusion module - combines multiple detection signals into dive events.

Phase 1: Audio-based detection only
Phase 2: Audio + motion-based validation
Phase 3: Audio + motion + person detection

Uses weighted combination of signals for improved accuracy.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DiveEvent:
    """Represents a detected dive event."""

    start_time: float
    """Start time in seconds (10s before splash)"""

    end_time: float
    """End time in seconds (3s after splash)"""

    splash_time: float
    """Exact splash time in seconds"""

    confidence: float
    """Confidence score 0.0-1.0"""

    audio_amplitude: float
    """Audio peak amplitude in dB"""

    motion_intensity: Optional[float] = None
    """Motion burst intensity (Phase 2+)"""

    had_person: bool = False
    """Person was in zone before dive (Phase 3+)"""

    # Metadata fields
    dive_number: Optional[int] = None
    """Sequential dive number"""

    notes: str = ""
    """Additional notes about the detection"""

    def __str__(self) -> str:
        return (
            f"Dive #{self.dive_number or '?'}: "
            f"{self.splash_time:.2f}s "
            f"(confidence {self.confidence:.1%})"
        )

    def duration(self) -> float:
        """Get duration of dive clip in seconds."""
        return self.end_time - self.start_time


def fuse_signals_audio_only(
    audio_peaks: List[tuple],
    pre_splash_buffer: float = 10.0,
    post_splash_buffer: float = 3.0,
) -> List[DiveEvent]:
    """
    Fuse signals into dive events (Phase 1: audio only).

    For Phase 1, we simply use audio peaks as direct dive indicators.
    Future phases will add validation from motion and person detection.

    Args:
        audio_peaks: List of (timestamp, amplitude) from detect_splash_peaks()
        pre_splash_buffer: Seconds before splash to include in clip
        post_splash_buffer: Seconds after splash to include in clip

    Returns:
        List of DiveEvent objects
    """
    if not audio_peaks:
        return []

    dives = []

    for idx, (splash_time, amplitude) in enumerate(audio_peaks, 1):
        # Phase 1: Confidence is based purely on audio amplitude
        # Higher amplitude = more confident it's a real splash
        # Normalize amplitude: assume -40dB to 0dB range
        normalized_amplitude = max(0.0, min(1.0, (amplitude + 40) / 40))

        # Phase 1 confidence: audio amplitude-based, with minimum threshold
        # Lower confidence in Phase 1 since we only have one signal
        confidence = 0.5 + (normalized_amplitude * 0.5)

        dive = DiveEvent(
            start_time=max(0.0, splash_time - pre_splash_buffer),
            end_time=splash_time + post_splash_buffer,
            splash_time=splash_time,
            confidence=confidence,
            audio_amplitude=amplitude,
            dive_number=idx,
        )

        dives.append(dive)

    return dives


def merge_overlapping_dives(
    dives: List[DiveEvent],
    min_gap: float = 5.0,
) -> List[DiveEvent]:
    """
    Merge dives that are too close together.

    When dives are detected very close together (< min_gap seconds),
    they're likely part of the same dive or back-to-back dives.

    Args:
        dives: List of DiveEvent objects
        min_gap: Minimum gap in seconds between dives (default 5s)

    Returns:
        Merged list of DiveEvent objects
    """
    if not dives:
        return []

    # Sort by start time
    dives = sorted(dives, key=lambda d: d.start_time)
    merged = [dives[0]]

    for dive in dives[1:]:
        prev = merged[-1]
        # Check if close enough to merge
        if dive.start_time < prev.end_time + min_gap:
            # Merge: extend previous dive and keep highest confidence
            prev.end_time = max(prev.end_time, dive.end_time)
            prev.confidence = max(prev.confidence, dive.confidence)
            prev.notes = f"merged with {dive.splash_time:.1f}s"
        else:
            merged.append(dive)

    # Re-number dives after merging
    for idx, dive in enumerate(merged, 1):
        dive.dive_number = idx

    return merged


def filter_dives_by_confidence(
    dives: List[DiveEvent],
    min_confidence: float = 0.5,
) -> List[DiveEvent]:
    """
    Filter dives by confidence threshold.

    Args:
        dives: List of DiveEvent objects
        min_confidence: Minimum confidence score (0.0-1.0)

    Returns:
        Filtered list of dives
    """
    return [d for d in dives if d.confidence >= min_confidence]


def fuse_signals_audio_motion(
    audio_peaks: List[Tuple[float, float]],
    motion_events: List[Tuple[float, float, float]],
    pre_splash_buffer: float = 10.0,
    post_splash_buffer: float = 3.0,
    motion_min_time_before: float = 2.0,
    motion_max_time_before: float = 12.0,
    confidence_audio_only: float = 0.3,
    confidence_audio_motion: float = 0.6,
) -> List[DiveEvent]:
    """
    Fuse audio and motion signals into dive events (Phase 2).

    Algorithm:
    1. For each audio peak (potential splash):
       - Search for motion activity 2-12 seconds before splash
       - If motion found nearby: confidence = 0.6 (audio + motion)
       - If motion NOT found: confidence = 0.3 (audio only, lower confidence)
    2. Return all detections with assigned confidence scores

    Args:
        audio_peaks: List of (timestamp, amplitude) from audio detection
        motion_events: List of (start, end, intensity) from motion detection
        pre_splash_buffer: Seconds before splash to include
        post_splash_buffer: Seconds after splash to include
        motion_min_time_before: Minimum time before splash to look for motion (seconds)
        motion_max_time_before: Maximum time before splash to look for motion (seconds)
        confidence_audio_only: Base confidence when only audio signal present
        confidence_audio_motion: Confidence when both audio and motion signals present

    Returns:
        List of DiveEvent objects with fused confidence scores

    Example:
        >>> audio_peaks = [(4.5, -12), (15.2, -18)]
        >>> motion_events = [(2.1, 4.8, 50), (12.5, 15.0, 45)]
        >>> dives = fuse_signals_audio_motion(audio_peaks, motion_events)
        >>> for dive in dives:
        ...     print(f"{dive}: confidence {dive.confidence:.1%}")
    """
    if not audio_peaks:
        return []

    dives = []

    for idx, (splash_time, amplitude) in enumerate(audio_peaks, 1):
        # Normalize audio amplitude (assume -40dB to 0dB range)
        normalized_amplitude = max(0.0, min(1.0, (amplitude + 40) / 40))

        # Check for motion before splash
        motion_match = None
        motion_intensity = None

        for motion_start, motion_end, intensity in motion_events:
            # Check if motion ends within the time window before splash
            time_before_splash = splash_time - motion_end

            if motion_min_time_before <= time_before_splash <= motion_max_time_before:
                motion_match = True
                motion_intensity = intensity
                break

        # Calculate confidence based on signal fusion
        if motion_match:
            # Audio + Motion: higher confidence
            confidence = confidence_audio_motion
            # Boost by normalized amplitude
            confidence = min(1.0, confidence + (normalized_amplitude * 0.2))
        else:
            # Audio only: lower confidence (Phase 1 level)
            confidence = confidence_audio_only
            confidence = 0.5 + (normalized_amplitude * 0.5)

        dive = DiveEvent(
            start_time=max(0.0, splash_time - pre_splash_buffer),
            end_time=splash_time + post_splash_buffer,
            splash_time=splash_time,
            confidence=confidence,
            audio_amplitude=amplitude,
            motion_intensity=motion_intensity,
            dive_number=idx,
            notes="audio+motion" if motion_match else "audio only",
        )

        dives.append(dive)

    return dives
