"""
Thumbnail generation module for dive preview.

Generates JPEG thumbnails (start, middle, end) for each dive without full video extraction.
This is 10-20x faster than extracting full MP4 clips.
"""

import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


@dataclass
class ThumbnailSet:
    """Set of thumbnails for a dive."""

    dive_id: int
    start_thumbnail: str  # Base64 encoded JPEG or file path
    middle_thumbnail: str
    end_thumbnail: str
    generated_at: float  # Timestamp
    file_paths: Optional[List[Path]] = None  # Actual file paths if saved


def generate_thumbnail_frame(
    video_path: str,
    timestamp: float,
    output_path: Optional[str] = None,
    width: int = 320,
    height: int = 180,
    quality: int = 85,
    as_base64: bool = False,
) -> str:
    """
    Extract a single frame from video as JPEG thumbnail.

    Args:
        video_path: Path to source video
        timestamp: Time in seconds to extract frame
        output_path: Where to save JPEG (if None, temp file)
        width: Thumbnail width in pixels
        height: Thumbnail height in pixels
        quality: JPEG quality 1-100
        as_base64: Return base64 encoded string instead of file path

    Returns:
        File path to JPEG or base64 encoded string

    Example:
        >>> generate_thumbnail_frame('video.mp4', 10.5, 'thumb.jpg')
        '/tmp/thumb.jpg'
    """
    video_path = str(video_path)

    # Create temp path if not provided
    if output_path is None:
        import tempfile
        fd, output_path = tempfile.mkstemp(suffix='.jpg', prefix='dive_thumb_')
        import os
        os.close(fd)  # Close file descriptor, FFmpeg will write to it

    output_path = str(output_path)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command to extract single frame
    # For 10-bit HEVC videos, use libswscale with detailed options
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite
        '-ss', str(timestamp),  # Seek to timestamp
        '-i', video_path,
        '-vframes', '1',  # Extract only 1 frame
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos',  # Resize with quality
        '-c:v', 'mjpeg',  # Explicitly use MJPEG encoder
        '-pix_fmt', 'yuvj420p',  # Use MJPEG-compatible YUV format
        '-q:v', str(100 - quality),  # Quality (FFmpeg uses inverted scale)
        '-sws_flags', 'lanczos',  # Use Lanczos scaling
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,  # 10 second timeout per frame
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg thumbnail generation failed: {result.stderr}")

        # Check file was created
        if not Path(output_path).exists():
            raise RuntimeError(f"Thumbnail file not created: {output_path}")

        # Return base64 if requested
        if as_base64:
            with open(output_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                # Clean up temp file if base64
                Path(output_path).unlink()
                return f"data:image/jpeg;base64,{encoded}"

        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Thumbnail generation timed out at {timestamp}s")
    except Exception as e:
        raise RuntimeError(f"Thumbnail generation failed: {str(e)}") from e


def generate_dive_thumbnails(
    video_path: str,
    dive_id: int,
    start_time: float,
    end_time: float,
    output_dir: Optional[str] = None,
    as_base64: bool = False,
) -> ThumbnailSet:
    """
    Generate 3 thumbnails for a dive: start, middle, end.

    Args:
        video_path: Path to source video
        dive_id: Dive identifier (for naming)
        start_time: Dive start time in seconds
        end_time: Dive end time in seconds
        output_dir: Directory to save thumbnails (optional)
        as_base64: Return base64 encoded strings instead of files

    Returns:
        ThumbnailSet with 3 thumbnails

    Example:
        >>> thumbs = generate_dive_thumbnails('video.mp4', 1, 10.0, 24.0)
        >>> print(thumbs.start_thumbnail)
        '/tmp/dive_001_start.jpg'
    """
    duration = end_time - start_time
    middle_time = start_time + (duration / 2)

    # Define output paths if saving to disk
    file_paths = []
    if output_dir and not as_base64:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_path = output_dir / f"dive_{dive_id:03d}_start.jpg"
        middle_path = output_dir / f"dive_{dive_id:03d}_middle.jpg"
        end_path = output_dir / f"dive_{dive_id:03d}_end.jpg"

        file_paths = [start_path, middle_path, end_path]
    else:
        start_path = middle_path = end_path = None

    # Generate thumbnails
    start_thumb = generate_thumbnail_frame(
        video_path, start_time, start_path, as_base64=as_base64
    )
    middle_thumb = generate_thumbnail_frame(
        video_path, middle_time, middle_path, as_base64=as_base64
    )
    end_thumb = generate_thumbnail_frame(
        video_path, end_time, end_path, as_base64=as_base64
    )

    return ThumbnailSet(
        dive_id=dive_id,
        start_thumbnail=start_thumb,
        middle_thumbnail=middle_thumb,
        end_thumbnail=end_thumb,
        generated_at=time.time(),
        file_paths=file_paths if file_paths else None,
    )


def generate_thumbnails_parallel(
    video_path: str,
    dives: List[Tuple[int, float, float]],  # (dive_id, start_time, end_time)
    output_dir: Optional[str] = None,
    max_workers: int = 4,
    as_base64: bool = False,
    progress_callback: Optional[callable] = None,
) -> List[ThumbnailSet]:
    """
    Generate thumbnails for multiple dives in parallel.

    Args:
        video_path: Path to source video
        dives: List of (dive_id, start_time, end_time) tuples
        output_dir: Directory to save thumbnails
        max_workers: Number of parallel workers
        as_base64: Return base64 encoded strings
        progress_callback: Function called with (completed, total) after each dive

    Returns:
        List of ThumbnailSet objects

    Example:
        >>> dives = [(1, 10.0, 24.0), (2, 30.0, 44.0), (3, 50.0, 64.0)]
        >>> thumbnails = generate_thumbnails_parallel('video.mp4', dives)
        >>> print(f"Generated {len(thumbnails)} thumbnail sets")
        Generated 3 thumbnail sets
    """
    results = []
    total = len(dives)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all thumbnail generation tasks
        future_to_dive = {
            executor.submit(
                generate_dive_thumbnails,
                video_path,
                dive_id,
                start_time,
                end_time,
                output_dir,
                as_base64,
            ): (dive_id, start_time, end_time)
            for dive_id, start_time, end_time in dives
        }

        # Collect results as they complete
        for future in as_completed(future_to_dive):
            dive_id, start_time, end_time = future_to_dive[future]

            try:
                thumbnail_set = future.result()
                results.append(thumbnail_set)
                completed += 1

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed, total)

            except Exception as e:
                print(f"Warning: Failed to generate thumbnails for dive {dive_id}: {e}")
                # Continue with other dives even if one fails
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

    # Sort by dive_id to maintain order
    results.sort(key=lambda x: x.dive_id)

    return results


def generate_timeline_thumbnails(
    video_path: str,
    start_time: float,
    end_time: float,
    num_frames: int = 8,
    width: int = 160,
    height: int = 90,
    as_base64: bool = True,
) -> List[str]:
    """
    Generate evenly-spaced timeline thumbnails across dive duration.

    Args:
        video_path: Path to source video
        start_time: Dive start time in seconds
        end_time: Dive end time in seconds
        num_frames: Number of frames to extract (default 8 = 2x4 grid)
        width: Thumbnail width in pixels
        height: Thumbnail height in pixels
        as_base64: Return base64 encoded strings

    Returns:
        List of base64 encoded thumbnail strings

    Example:
        >>> frames = generate_timeline_thumbnails('video.mp4', 10.0, 24.0, 8)
        >>> len(frames)
        8
    """
    thumbnails = []
    duration = end_time - start_time

    for i in range(num_frames):
        # Distribute frames evenly across the dive duration
        progress = i / (num_frames - 1) if num_frames > 1 else 0
        timestamp = start_time + (progress * duration)

        try:
            thumb = generate_thumbnail_frame(
                video_path,
                timestamp,
                width=width,
                height=height,
                quality=70,  # Lower quality for timeline
                as_base64=as_base64,
            )
            thumbnails.append(thumb)
        except Exception as e:
            # Return placeholder if generation fails
            thumbnails.append(None)

    return thumbnails


def cleanup_thumbnails(thumbnail_sets: List[ThumbnailSet]):
    """
    Delete thumbnail files from disk.

    Args:
        thumbnail_sets: List of ThumbnailSet objects with file_paths
    """
    for thumb_set in thumbnail_sets:
        if thumb_set.file_paths:
            for path in thumb_set.file_paths:
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except Exception as e:
                        print(f"Warning: Failed to delete thumbnail {path}: {e}")
