"""Detection modules for dive identification."""

from .audio import detect_audio_peaks, extract_audio
from .fusion import fuse_signals, DiveEvent

__all__ = ["detect_audio_peaks", "extract_audio", "fuse_signals", "DiveEvent"]
