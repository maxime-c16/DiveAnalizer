"""
DiveAnalyzer v2.0 - Automated diving video clip extraction

Multi-modal detection using audio peaks, motion analysis, and person detection.

Phase 1: Audio-based detection
Phase 2: Motion-based validation
Phase 3: Person detection integration

Import modules individually to avoid dependency issues:
  - from diveanalyzer.detection.fusion import DiveEvent
  - from diveanalyzer.detection.motion import detect_motion_bursts
  - from diveanalyzer.storage.cache import CacheManager
"""

__version__ = "2.0.0"

# Core types exported (don't require audio module)
try:
    from .detection.fusion import DiveEvent
except ImportError:
    pass

from .extraction.ffmpeg import extract_dive_clip

__all__ = [
    "DiveEvent",
    "extract_dive_clip",
]
