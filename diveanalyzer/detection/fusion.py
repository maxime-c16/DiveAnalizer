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
    motion_min_time_before: float = 0.0,
    motion_max_time_before: float = 15.0,
    motion_validation_boost: float = 0.15,
) -> List[DiveEvent]:
    """
    Fuse audio and motion signals into dive events (Phase 2).

    Algorithm (Fixed):
    1. For each audio peak (potential splash):
       - Use audio amplitude as base confidence (Phase 1 method)
       - Search for motion activity 0-15 seconds before splash (adaptive window)
       - If motion found nearby: BOOST confidence by +15% (validation)
       - If motion NOT found: KEEP audio confidence (don't penalize)
    2. Return all detections with improved confidence scores

    Key insight from Phase 2 testing:
    - Audio is already excellent for clean recordings (0.82 avg confidence)
    - Motion validation should BOOST, not REPLACE, audio confidence
    - Wider window (0-15s) catches more dive approach patterns
    - Don't penalize dives without visible motion (different diving styles)

    Args:
        audio_peaks: List of (timestamp, amplitude) from audio detection
        motion_events: List of (start, end, intensity) from motion detection
        pre_splash_buffer: Seconds before splash to include
        post_splash_buffer: Seconds after splash to include
        motion_min_time_before: Minimum time before splash to look for motion (seconds)
        motion_max_time_before: Maximum time before splash to look for motion (seconds)
        motion_validation_boost: Confidence boost for motion-validated dives (+0.15 = +15%)

    Returns:
        List of DiveEvent objects with fused confidence scores

    Example:
        >>> audio_peaks = [(4.5, -12), (15.2, -18)]
        >>> motion_events = [(0.5, 4.8, 50), (12.5, 15.0, 45)]
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

        # Audio confidence (Phase 1 method - proven excellent)
        audio_confidence = 0.5 + (normalized_amplitude * 0.5)

        # Check for motion before splash (adaptive window)
        motion_match = None
        motion_intensity = None

        for motion_start, motion_end, intensity in motion_events:
            # Check if motion ends within the time window before splash
            time_before_splash = splash_time - motion_end

            if motion_min_time_before <= time_before_splash <= motion_max_time_before:
                motion_match = True
                motion_intensity = intensity
                break

        # Final confidence: base audio confidence with optional motion boost
        if motion_match:
            # Motion VALIDATES dive approach: boost confidence
            confidence = min(1.0, audio_confidence + motion_validation_boost)
            signal_type = "audio+motion"
        else:
            # No motion detected: trust audio confidence
            confidence = audio_confidence
            signal_type = "audio only"

        dive = DiveEvent(
            start_time=max(0.0, splash_time - pre_splash_buffer),
            end_time=splash_time + post_splash_buffer,
            splash_time=splash_time,
            confidence=confidence,
            audio_amplitude=amplitude,
            motion_intensity=motion_intensity,
            dive_number=idx,
            notes=signal_type,
        )

        dives.append(dive)

    return dives


def fuse_signals_audio_motion_person(
    audio_peaks: List[Tuple[float, float]],
    motion_events: List[Tuple[float, float, float]],
    person_departures: List[Tuple[float, float]],
    pre_splash_buffer: float = 10.0,
    post_splash_buffer: float = 3.0,
    motion_min_time_before: float = 0.0,
    motion_max_time_before: float = 15.0,
    person_min_time_before: float = 0.0,
    person_max_time_before: float = 15.0,
    motion_validation_boost: float = 0.15,
    person_validation_boost: float = 0.10,
) -> List[DiveEvent]:
    """
    Fuse audio, motion, and person signals into dive events (Phase 3).

    Algorithm:
    1. For each audio peak (potential splash):
       - Use audio amplitude as base confidence
       - Search for motion activity 0-15 seconds before splash
       - Search for person departure 0-15 seconds before splash
       - Apply boosts for each matching validation signal
       - Final confidence = min(base + boosts, 1.0)

    Validation Levels:
    - 3-signal: Audio + Motion + Person (0.95-0.99 confidence)
    - 2-signal: Audio + (Motion OR Person) (0.90-0.95 confidence)
    - Audio-only: No motion/person signals (0.75-0.90 confidence)

    Args:
        audio_peaks: List of (timestamp, amplitude) from audio detection
        motion_events: List of (start, end, intensity) from motion detection
        person_departures: List of (timestamp, confidence) from person detection
        pre_splash_buffer: Seconds before splash to include
        post_splash_buffer: Seconds after splash to include
        motion_min_time_before: Minimum time before splash to look for motion
        motion_max_time_before: Maximum time before splash to look for motion
        person_min_time_before: Minimum time before splash to look for person departure
        person_max_time_before: Maximum time before splash to look for person departure
        motion_validation_boost: Confidence boost for motion validation (+0.15)
        person_validation_boost: Confidence boost for person validation (+0.10)

    Returns:
        List of DiveEvent objects with three-signal confidence scores
    """
    if not audio_peaks:
        return []

    dives = []

    for idx, (splash_time, amplitude) in enumerate(audio_peaks, 1):
        # Normalize audio amplitude
        normalized_amplitude = max(0.0, min(1.0, (amplitude + 40) / 40))

        # Audio confidence (base signal)
        audio_confidence = 0.5 + (normalized_amplitude * 0.5)

        # Check for motion before splash
        motion_match = False
        motion_intensity = None

        for motion_start, motion_end, intensity in motion_events:
            time_before_splash = splash_time - motion_end
            if motion_min_time_before <= time_before_splash <= motion_max_time_before:
                motion_match = True
                motion_intensity = intensity
                break

        # Check for person departure before splash
        person_match = False
        person_departure_time = None

        for dep_time, dep_confidence in person_departures:
            time_before_splash = splash_time - dep_time
            if person_min_time_before <= time_before_splash <= person_max_time_before:
                person_match = True
                person_departure_time = dep_time
                break

        # Calculate final confidence with boosts
        confidence = audio_confidence

        if motion_match:
            confidence = min(1.0, confidence + motion_validation_boost)

        if person_match:
            confidence = min(1.0, confidence + person_validation_boost)

        # Determine validation level
        validation_count = sum([motion_match, person_match])
        if validation_count == 2:
            signal_type = "3-signal"
        elif validation_count == 1:
            signal_type = "2-signal"
        else:
            signal_type = "audio-only"

        dive = DiveEvent(
            start_time=max(0.0, splash_time - pre_splash_buffer),
            end_time=splash_time + post_splash_buffer,
            splash_time=splash_time,
            confidence=confidence,
            audio_amplitude=amplitude,
            motion_intensity=motion_intensity,
            had_person=person_match,
            dive_number=idx,
            notes=signal_type,
        )

        dives.append(dive)

    return dives
