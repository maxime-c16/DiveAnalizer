"""Video extraction modules."""

from .ffmpeg import extract_dive_clip, extract_multiple_dives, get_video_duration
from .proxy import generate_proxy, get_video_resolution, get_proxy_size_reduction

__all__ = [
    "extract_dive_clip",
    "extract_multiple_dives",
    "get_video_duration",
    "generate_proxy",
    "get_video_resolution",
    "get_proxy_size_reduction",
]
