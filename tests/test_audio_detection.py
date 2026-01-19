"""
Tests for audio detection module.

Tests can be run with: pytest tests/test_audio_detection.py -v
"""

import tempfile
from pathlib import Path
import numpy as np

# Try importing, but tests will be skipped if dependencies aren't installed
try:
    from diveanalyzer.detection.audio import (
        detect_splash_peaks,
        filter_splash_peaks,
        get_audio_properties,
    )
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


def create_synthetic_audio(duration_sec: float, splash_times: list) -> tuple:
    """
    Create synthetic audio with known splash times.

    Returns: (audio_array, sample_rate)
    """
    sr = 22050
    t = np.linspace(0, duration_sec, int(sr * duration_sec))

    # Start with noise
    audio = np.random.normal(0, 0.01, len(t))

    # Add loud peaks at splash times
    for splash_time in splash_times:
        # Find sample index
        idx = int(splash_time * sr)
        if 0 <= idx < len(audio):
            # Add transient peak
            width = int(0.1 * sr)  # 100ms
            start = max(0, idx - width // 2)
            end = min(len(audio), idx + width // 2)
            peak_envelope = np.exp(-((np.arange(end - start) - width // 2) ** 2) / 100)
            audio[start:end] += 0.5 * peak_envelope

    return audio, sr


if AUDIO_AVAILABLE:

    def test_detect_splash_peaks_with_synthetic():
        """Test splash detection on synthetic audio."""
        # Create audio with 3 known splashes
        splash_times = [5.0, 15.0, 25.0]
        audio, sr = create_synthetic_audio(30.0, splash_times)

        # Save to temp file
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            temp_path = tmp.name

        try:
            peaks = detect_splash_peaks(temp_path, threshold_db=-30.0, min_distance_sec=5.0)

            # Should detect approximately 3 peaks
            assert len(peaks) >= 2, f"Expected at least 2 peaks, got {len(peaks)}"
            assert len(peaks) <= 4, f"Expected at most 4 peaks, got {len(peaks)}"

        finally:
            Path(temp_path).unlink()

    def test_filter_splash_peaks():
        """Test peak filtering by amplitude."""
        peaks = [(1.0, -25.0), (2.0, -15.0), (3.0, -5.0)]

        filtered = filter_splash_peaks(peaks, min_amplitude_db=-20.0)

        assert len(filtered) == 2
        assert filtered[0] == (2.0, -15.0)
        assert filtered[1] == (3.0, -5.0)

    def test_detect_splash_peaks_no_splashes():
        """Test detection on silent audio."""
        audio = np.random.normal(0, 0.001, 22050)  # Very quiet noise
        sr = 22050

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            temp_path = tmp.name

        try:
            peaks = detect_splash_peaks(temp_path, threshold_db=-20.0)
            # Should detect very few or no peaks in quiet audio
            assert len(peaks) == 0, f"Expected 0 peaks in quiet audio, got {len(peaks)}"

        finally:
            Path(temp_path).unlink()

else:

    def test_skip_audio_detection():
        """Placeholder test when audio dependencies not available."""
        assert True
