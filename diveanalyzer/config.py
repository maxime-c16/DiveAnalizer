"""
Configuration management for DiveAnalyzer.

Handles settings, defaults, and environment variables.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DetectionConfig:
    """Configuration for dive detection."""

    # Audio detection parameters
    audio_threshold_db: float = -25.0
    """Audio peak height threshold in dB (relative to max)"""

    audio_min_distance_sec: float = 5.0
    """Minimum time between consecutive splash detections (seconds)"""

    audio_prominence: float = 5.0
    """Minimum audio peak prominence in dB"""

    # Dive clip extraction parameters
    pre_splash_buffer: float = 10.0
    """Seconds to include before splash time"""

    post_splash_buffer: float = 3.0
    """Seconds to include after splash time"""

    # Filtering parameters
    min_confidence: float = 0.5
    """Minimum confidence to include dive (0.0-1.0)"""

    min_gap_sec: float = 5.0
    """Minimum gap between dives for merging (seconds)"""

    # Processing parameters
    enable_audio_cache: bool = True
    """Cache extracted audio for reuse"""

    enable_proxy_cache: bool = True
    """Cache proxy video generation (Phase 2+)"""

    # Motion detection parameters (Phase 2)
    motion_enabled: bool = True
    """Enable motion-based validation for dives"""

    motion_sample_fps: float = 5.0
    """Frames per second to sample for motion analysis"""

    motion_threshold_percentile: float = 80.0
    """Percentile above which motion triggers burst"""

    motion_min_burst_duration: float = 0.5
    """Minimum motion burst duration in seconds"""

    motion_min_time_before_splash: float = 2.0
    """Minimum time before splash to look for motion"""

    motion_max_time_before_splash: float = 12.0
    """Maximum time before splash to look for motion"""

    # Proxy parameters (Phase 2)
    proxy_height: int = 480
    """Proxy video height in pixels"""

    proxy_preset: str = "ultrafast"
    """FFmpeg encoding preset (ultrafast, superfast, veryfast, faster)"""

    proxy_crf: int = 28
    """Quality setting (0-51, lower = better)"""

    proxy_min_video_size_mb: float = 500.0
    """Only generate proxy for videos larger than this"""

    # Signal fusion parameters (Phase 2)
    confidence_audio_only: float = 0.3
    """Confidence for audio-only detection"""

    confidence_audio_motion: float = 0.6
    """Confidence for audio + motion detection"""

    confidence_all_signals: float = 0.9
    """Confidence for audio + motion + person detection"""

    # Output parameters
    preserve_audio: bool = True
    """Include audio in extracted clips"""

    output_filename_pattern: str = "dive_{number:03d}.mp4"
    """Filename pattern for extracted clips"""

    verbose: bool = False
    """Print detailed processing information"""

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DetectionConfig":
        """Create config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class CacheConfig:
    """Configuration for local caching."""

    cache_dir: Path = None
    """Cache directory (default: ~/.diveanalyzer)"""

    cache_max_age_days: int = 7
    """Maximum age of cached files before cleanup"""

    enable_cleanup: bool = True
    """Automatically clean old cache files"""

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".diveanalyzer"

        # Ensure subdirectories exist
        self.audio_dir = self.cache_dir / "audio"
        self.proxy_dir = self.cache_dir / "proxies"
        self.metadata_dir = self.cache_dir / "metadata"

        for directory in [self.audio_dir, self.proxy_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class iCloudConfig:
    """Configuration for iCloud integration."""

    enabled: bool = False
    """Enable iCloud Drive integration"""

    local_mount: Path = None
    """Local path to iCloud Drive mount (auto-detected if None)"""

    folder_name: str = "Diving"
    """Folder name in iCloud Drive to search for videos"""

    @property
    def icloud_path(self) -> Optional[Path]:
        """Get iCloud Drive path on macOS."""
        if self.local_mount:
            return self.local_mount

        # Try default macOS iCloud path
        icloud_default = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"
        if icloud_default.exists():
            return icloud_default

        return None


class Config:
    """Main configuration class."""

    def __init__(self):
        self.detection = DetectionConfig()
        self.cache = CacheConfig()
        self.icloud = iCloudConfig()

    @classmethod
    def from_file(cls, config_file: Path) -> "Config":
        """Load configuration from file (YAML/JSON)."""
        import json

        config = cls()
        if config_file.exists():
            with open(config_file) as f:
                data = json.load(f)
                if "detection" in data:
                    config.detection = DetectionConfig.from_dict(data["detection"])
        return config

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "detection": vars(self.detection),
            "cache": {
                "cache_dir": str(self.cache.cache_dir),
                "cache_max_age_days": self.cache.cache_max_age_days,
            },
            "icloud": {
                "enabled": self.icloud.enabled,
                "folder_name": self.icloud.folder_name,
            },
        }


# Global config instance
_config = None


def get_config() -> Config:
    """Get or create global config."""
    global _config
    if _config is None:
        _config = Config()
    return _config
