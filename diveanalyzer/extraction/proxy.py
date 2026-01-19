"""Proxy video generation for efficient detection and caching.

Generates 480p proxy videos (~10x smaller than original 4K) for:
- Faster motion detection
- Faster person detection
- Reduced memory usage
- Local caching

Original video is used only for final clip extraction to preserve quality.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, Tuple
from .ffmpeg import get_video_duration


def get_video_resolution(video_path: Union[str, Path]) -> Tuple[int, int]:
    """Get video resolution (width, height) using FFprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height) in pixels

    Raises:
        RuntimeError: If FFprobe fails or video is invalid
    """
    video_path = str(video_path)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split("x"))
        return width, height
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get video resolution: {e}") from e


def generate_proxy(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    height: int = 480,
    preset: str = "ultrafast",
    crf: int = 28,
    verbose: bool = False,
) -> None:
    """Generate a 480p proxy video for faster detection.

    Args:
        video_path: Path to source video
        output_path: Path to output proxy video
        height: Proxy height in pixels (default 480p)
        preset: FFmpeg encoding preset (ultrafast, superfast, veryfast, faster)
        crf: Quality (0-51, lower = better, 28 = acceptable for detection)
        verbose: Print FFmpeg output

    Raises:
        RuntimeError: If FFmpeg encoding fails
        subprocess.CalledProcessError: If FFmpeg fails

    Example:
        >>> generate_proxy('4k_video.mov', 'proxy_480p.mp4', height=480)
    """
    video_path = str(video_path)
    output_path = str(output_path)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    # Scale to height maintaining aspect ratio
    # -2 means: divisible by 2 for codec compatibility
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-i", video_path,
        "-vf", f"scale=-2:{height}",  # Scale to target height
        "-c:v", "libx264",  # H.264 codec (best compatibility)
        "-preset", preset,  # Speed vs quality trade-off
        "-crf", str(crf),  # Quality (lower = better)
        "-c:a", "aac",  # Audio codec
        "-b:a", "64k",  # Lower bitrate for proxy
        "-y",  # Overwrite
        output_path,
    ]

    try:
        if verbose:
            print(f"Generating proxy: {Path(video_path).name} → {height}p")

        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr if hasattr(result, "stderr") else "Unknown error"
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=error_msg)

        if verbose:
            output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            original_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            ratio = output_size_mb / original_size_mb
            print(f"  ✓ Proxy created ({original_size_mb:.0f}MB → {output_size_mb:.0f}MB, {ratio:.1%})")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg proxy generation failed: {e.stderr if hasattr(e, 'stderr') else str(e)}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to generate proxy: {e}") from e


def get_proxy_size_reduction(
    video_path: Union[str, Path],
    proxy_height: int = 480,
) -> dict:
    """Estimate proxy file size reduction before generation.

    Args:
        video_path: Path to source video
        proxy_height: Target proxy height in pixels

    Returns:
        Dictionary with size estimates
    """
    video_path = Path(video_path)
    original_size_mb = video_path.stat().st_size / (1024 * 1024)

    try:
        width, height = get_video_resolution(video_path)
        aspect_ratio = width / height if height > 0 else 1.0
        proxy_width = int(proxy_height * aspect_ratio)

        # Rough estimate: size scales with pixel count
        # Plus lower bitrate audio for proxy
        scale_factor = (proxy_width * proxy_height) / (width * height)
        estimated_proxy_mb = original_size_mb * scale_factor * 1.1  # 1.1x for audio

        return {
            "original_resolution": (width, height),
            "proxy_resolution": (proxy_width, proxy_height),
            "original_size_mb": original_size_mb,
            "estimated_proxy_size_mb": estimated_proxy_mb,
            "estimated_reduction_percent": 100 * (1 - estimated_proxy_mb / original_size_mb),
        }
    except Exception as e:
        return {
            "error": str(e),
            "original_size_mb": original_size_mb,
        }


def optimize_proxy_settings(video_duration: float) -> dict:
    """Get optimized proxy settings based on video duration.

    For longer videos, use faster presets to reduce proxy generation time.

    Args:
        video_duration: Video duration in seconds

    Returns:
        Dictionary with recommended proxy generation settings
    """
    if video_duration < 60:  # < 1 minute
        return {"preset": "faster", "crf": 26, "height": 480}
    elif video_duration < 600:  # < 10 minutes
        return {"preset": "ultrafast", "crf": 28, "height": 480}
    else:  # > 10 minutes
        return {"preset": "ultrafast", "crf": 30, "height": 360}


def is_proxy_generation_needed(
    video_path: Union[str, Path],
    size_threshold_mb: float = 500,
) -> bool:
    """Check if proxy generation is recommended for this video.

    Args:
        video_path: Path to video file
        size_threshold_mb: Videos larger than this benefit from proxy

    Returns:
        True if proxy is recommended, False if video is already small
    """
    file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
    return file_size_mb > size_threshold_mb
