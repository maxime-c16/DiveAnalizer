"""
DiveAnalyzer v2.0 - Automated diving video clip extraction

Multi-modal detection using audio peaks, motion analysis, and person detection.
"""

__version__ = "2.0.0"

from .detection.audio import detect_audio_peaks, extract_audio
from .detection.fusion import fuse_signals, DiveEvent
from .extraction.ffmpeg import extract_dive_clip

__all__ = [
    "detect_audio_peaks",
    "extract_audio",
    "fuse_signals",
    "DiveEvent",
    "extract_dive_clip",
]
