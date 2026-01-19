"""Detection modules for dive identification.

Audio module (requires librosa) is imported on-demand.
Other modules (motion, fusion) are imported directly.
"""

# Audio detection (Phase 1) - imported on-demand due to librosa dependency
# from .audio import detect_audio_peaks, extract_audio

# Motion detection (Phase 2) - no external dependencies
from .motion import detect_motion_bursts, find_motion_before_event

# Signal fusion - no external dependencies
from .fusion import (
    fuse_signals_audio_only,
    fuse_signals_audio_motion,
    DiveEvent,
    merge_overlapping_dives,
    filter_dives_by_confidence,
)

__all__ = [
    # Audio detection (import manually: from diveanalyzer.detection.audio import ...)
    # "detect_audio_peaks",
    # "extract_audio",
    # Motion detection (Phase 2)
    "detect_motion_bursts",
    "find_motion_before_event",
    # Signal fusion
    "fuse_signals_audio_only",
    "fuse_signals_audio_motion",
    "DiveEvent",
    "merge_overlapping_dives",
    "filter_dives_by_confidence",
]
