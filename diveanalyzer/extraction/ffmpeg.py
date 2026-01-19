"""
Video extraction module using FFmpeg.

Uses FFmpeg stream copy for instant extraction without re-encoding.
This is orders of magnitude faster than re-encoding with OpenCV/ffmpeg.
"""

import subprocess
from pathlib import Path
from typing import Optional, Union
from datetime import timedelta


def extract_dive_clip(
    video_path: Union[str, Path],
    start_time: float,
    end_time: float,
    output_path: Union[str, Path],
    audio_enabled: bool = True,
    verbose: bool = False,
) -> bool:
    """
    Extract a dive clip from video using FFmpeg stream copy.

    Stream copy (-c copy) means:
    - No re-encoding
    - Instant processing (only file copy)
    - Perfect quality preservation
    - 100x faster than transcoding

    Args:
        video_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to output clip
        audio_enabled: Include audio track (default True)
        verbose: Print FFmpeg output

    Returns:
        True if successful, False otherwise

    Example:
        >>> extract_dive_clip('session.mp4', 10.5, 24.3, 'dive_1.mp4')
        True
    """
    video_path = str(video_path)
    output_path = str(output_path)

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = end_time - start_time

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-ss",
        str(start_time),  # Seek position BEFORE input (fast)
        "-i",
        video_path,
        "-t",
        str(duration),  # Duration to extract
        "-c",
        "copy",  # Stream copy (no re-encoding!)
        "-avoid_negative_ts",
        "make_zero",  # Normalize timestamps
    ]

    if not audio_enabled:
        cmd.extend(["-an"])  # Remove audio

    cmd.append(output_path)

    try:
        stderr_pipe = subprocess.DEVNULL
        stdout_pipe = subprocess.DEVNULL

        if verbose:
            stderr_pipe = None
            stdout_pipe = None

        result = subprocess.run(
            cmd,
            capture_output=(not verbose),
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr if hasattr(result, "stderr") else "Unknown error"
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=error_msg)

        return True

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg extraction failed: {e.stderr if hasattr(e, 'stderr') else str(e)}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to extract dive clip: {e}") from e


def extract_multiple_dives(
    video_path: Union[str, Path],
    dives: list,
    output_dir: Union[str, Path],
    audio_enabled: bool = True,
    name_pattern: str = "dive_{number:03d}.mp4",
    verbose: bool = False,
) -> dict:
    """
    Extract multiple dive clips from a single video.

    Args:
        video_path: Path to source video
        dives: List of DiveEvent objects with start_time, end_time, dive_number
        output_dir: Directory to save extracted clips
        audio_enabled: Include audio in extracts
        name_pattern: Filename pattern with {number} placeholder
        verbose: Print progress

    Returns:
        Dictionary with results: {dive_number: (success, output_path, error)}

    Example:
        >>> from diveanalyzer import detect_audio_peaks, fuse_signals_audio_only
        >>> peaks = detect_audio_peaks('session.mp4')
        >>> dives = fuse_signals_audio_only(peaks)
        >>> results = extract_multiple_dives('session.mp4', dives, './output')
        >>> for dive_num, (success, path, error) in results.items():
        ...     if success:
        ...         print(f"Extracted dive {dive_num}: {path}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for dive in dives:
        # Generate output filename
        filename = name_pattern.format(number=dive.dive_number)
        output_path = output_dir / filename

        try:
            if verbose:
                print(
                    f"Extracting dive {dive.dive_number}: "
                    f"{dive.start_time:.1f}s - {dive.end_time:.1f}s"
                )

            success = extract_dive_clip(
                video_path,
                dive.start_time,
                dive.end_time,
                output_path,
                audio_enabled=audio_enabled,
                verbose=verbose,
            )

            results[dive.dive_number] = (success, str(output_path), None)

        except Exception as e:
            results[dive.dive_number] = (False, None, str(e))
            if verbose:
                print(f"  ✗ Failed: {e}")

    return results


def get_video_duration(video_path: Union[str, Path]) -> float:
    """
    Get video duration using FFmpeg.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If FFmpeg fails or video is invalid
    """
    video_path = str(video_path)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1:noprint_wrappers=1",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get video duration: {e}") from e


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    td = timedelta(seconds=seconds)
    return str(td).split(".")[0]  # Remove microseconds
