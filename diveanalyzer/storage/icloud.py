"""iCloud Drive integration for DiveAnalyzer.

Provides utilities to find diving videos in iCloud Drive on macOS.
Works with native macOS iCloud Drive mounting (recommended).

Alternative options documented in ARCHITECTURE_PLAN.md:
- Option B: pyicloud API for cross-platform
- Option C: rclone mount for power users
"""

from pathlib import Path
from typing import List, Optional


def get_icloud_path() -> Optional[Path]:
    """Get macOS iCloud Drive path if it exists and is accessible.

    macOS automatically syncs iCloud Drive to:
    ~/Library/Mobile Documents/com~apple~CloudDocs/

    Returns:
        Path to iCloud Drive, or None if not available
    """
    icloud_path = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"

    if icloud_path.exists() and icloud_path.is_dir():
        return icloud_path

    return None


def find_icloud_videos(
    folder_name: str = "Diving",
    video_extensions: tuple = (".mov", ".mp4", ".m4v", ".mkv"),
) -> List[Path]:
    """Find all diving videos in iCloud Drive.

    Args:
        folder_name: Folder name in iCloud Drive (default: "Diving")
        video_extensions: Tuple of video file extensions to search for

    Returns:
        List of Path objects for video files found

    Example:
        >>> videos = find_icloud_videos("Diving")
        >>> for v in videos:
        ...     print(v.name)
    """
    icloud_path = get_icloud_path()
    if not icloud_path:
        return []

    diving_folder = icloud_path / folder_name
    if not diving_folder.exists():
        return []

    videos = []
    for ext in video_extensions:
        videos.extend(diving_folder.glob(f"**/*{ext}"))

    return sorted(videos)


def find_recent_icloud_videos(
    folder_name: str = "Diving",
    max_age_hours: int = 24,
    video_extensions: tuple = (".mov", ".mp4", ".m4v", ".mkv"),
) -> List[Path]:
    """Find recently added diving videos in iCloud Drive.

    Args:
        folder_name: Folder name in iCloud Drive
        max_age_hours: Only return videos newer than this (in hours)
        video_extensions: Tuple of video file extensions to search for

    Returns:
        List of recently modified video files, sorted by modification time (newest first)
    """
    from datetime import datetime, timedelta
    import time

    videos = find_icloud_videos(folder_name, video_extensions)
    if not videos:
        return []

    cutoff_time = time.time() - (max_age_hours * 3600)
    recent = [
        v for v in videos
        if v.stat().st_mtime > cutoff_time
    ]

    # Sort by modification time (newest first)
    recent.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return recent


def is_icloud_available() -> bool:
    """Check if iCloud Drive is available and accessible.

    Returns:
        True if iCloud Drive is mounted and accessible
    """
    return get_icloud_path() is not None
