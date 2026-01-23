"""Video extraction modules."""

from .ffmpeg import extract_dive_clip, extract_multiple_dives, get_video_duration
from .proxy import generate_proxy, get_video_resolution, get_proxy_size_reduction
from .thumbnails import (
    generate_thumbnail_frame,
    generate_dive_thumbnails,
    generate_thumbnails_parallel,
    cleanup_thumbnails,
    ThumbnailSet,
)

__all__ = [
    "extract_dive_clip",
    "extract_multiple_dives",
    "get_video_duration",
    "generate_proxy",
    "get_video_resolution",
    "get_proxy_size_reduction",
    "generate_thumbnail_frame",
    "generate_dive_thumbnails",
    "generate_thumbnails_parallel",
    "cleanup_thumbnails",
    "ThumbnailSet",
]
