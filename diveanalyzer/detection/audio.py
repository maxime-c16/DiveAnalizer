"""
Audio-based splash detection module.

Detects dive splash events by analyzing audio peaks.
This is the primary signal for dive detection.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import librosa
from scipy.signal import find_peaks


def extract_audio(video_path: str, output_format: str = "wav") -> str:
    """
    Extract audio track from video using FFmpeg.

    Args:
        video_path: Path to input video file
        output_format: Audio format (wav, mp3, etc.)

    Returns:
        Path to extracted audio file

    Raises:
        subprocess.CalledProcessError: If FFmpeg extraction fails
    """
    video_path = str(video_path)

    # Create temporary file for audio
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
        audio_path = tmp.name

    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite without asking
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # WAV PCM codec
            "-ar",
            "22050",  # Sample rate (librosa standard)
            "-ac",
            "1",  # Mono
            audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, stderr=result.stderr
            )

        return audio_path

    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to extract audio: {e}") from e


def detect_splash_peaks(
    audio_path: str,
    threshold_db: float = -25.0,
    min_distance_sec: float = 5.0,
    prominence: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Detect splash sounds in audio by finding peaks.

    This is the primary detection signal. Splash sounds have:
    - Sharp transient onset (< 100ms rise time)
    - Distinctive frequency content (100-2000 Hz)
    - High amplitude relative to background

    Args:
        audio_path: Path to audio file
        threshold_db: Minimum height of peak in dB (relative to max)
        min_distance_sec: Minimum time between peaks in seconds
        prominence: Minimum prominence of peak (dB above surroundings)

    Returns:
        List of (timestamp_sec, amplitude_db) tuples for detected splashes

    Example:
        >>> peaks = detect_splash_peaks('audio.wav')
        >>> for time, amplitude in peaks:
        ...     print(f"Splash at {time:.1f}s, amplitude {amplitude:.1f}dB")
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)

        # Compute RMS energy in short windows (~23ms)
        # This provides temporal resolution for detecting transient splashes
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Calculate minimum distance in frames
        min_distance_frames = int(min_distance_sec * sr / hop_length)

        # Find peaks using scipy
        peaks, properties = find_peaks(
            rms_db,
            height=threshold_db,
            distance=min_distance_frames,
            prominence=prominence,
        )

        # Convert frame indices to timestamps
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        peak_amplitudes = rms_db[peaks]

        # Return as list of (time, amplitude) tuples
        results = list(zip(peak_times, peak_amplitudes))

        # Sort by time (should already be sorted, but be safe)
        results = sorted(results, key=lambda x: x[0])

        return results

    except Exception as e:
        raise RuntimeError(f"Failed to detect splash peaks: {e}") from e


def filter_splash_peaks(
    peaks: List[Tuple[float, float]],
    min_amplitude_db: float = -20.0,
) -> List[Tuple[float, float]]:
    """
    Filter splash peaks by amplitude threshold.

    Args:
        peaks: List of (timestamp, amplitude) from detect_splash_peaks
        min_amplitude_db: Minimum amplitude to keep

    Returns:
        Filtered list of peaks
    """
    return [(t, a) for t, a in peaks if a >= min_amplitude_db]


def get_audio_properties(audio_path: str) -> dict:
    """
    Get properties of audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with sample_rate, duration, channels
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        # Assume mono, but could check channels
        return {
            "sample_rate": sr,
            "duration": duration,
            "channels": 1,
            "total_samples": len(y),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get audio properties: {e}") from e
