"""Cache management system for DiveAnalyzer.

Handles caching of:
- Audio tracks (extracted once from video)
- Proxy videos (480p, 10x smaller)
- Detection metadata (JSON results)

Cache is stored in ~/.diveanalyzer/ with automatic cleanup after 7 days.
"""

import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


def get_cache_dir() -> Path:
    """Get or create cache directory."""
    cache_dir = Path.home() / ".diveanalyzer" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class CacheEntry:
    """Metadata about a cached file."""
    hash: str
    source_path: str
    source_size: int
    source_mtime: float
    cache_type: str  # "audio", "proxy", "metadata"
    created_at: str
    expires_at: str

    def is_expired(self) -> bool:
        """Check if cache entry has expired (7 days)."""
        expires = datetime.fromisoformat(self.expires_at)
        return datetime.now() > expires


class CacheManager:
    """Manages cache for audio, proxies, and metadata."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager.

        Args:
            cache_dir: Optional custom cache directory (defaults to ~/.diveanalyzer/cache)
        """
        self.cache_dir = cache_dir or get_cache_dir()
        self.audio_dir = self.cache_dir / "audio"
        self.proxy_dir = self.cache_dir / "proxies"
        self.metadata_dir = self.cache_dir / "metadata"
        self.index_file = self.cache_dir / "index.json"

        # Create subdirectories
        for d in [self.audio_dir, self.proxy_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self.index = self._load_index()

    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash based on file path, size, and modification time."""
        p = Path(file_path)
        stat = p.stat()
        key = f"{p.name}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _load_index(self) -> Dict[str, CacheEntry]:
        """Load cache index from disk."""
        if not self.index_file.exists():
            return {}

        try:
            with open(self.index_file) as f:
                data = json.load(f)
            return {
                k: CacheEntry(**v) for k, v in data.items()
            }
        except Exception:
            return {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        data = {k: asdict(v) for k, v in self.index.items()}
        with open(self.index_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_audio(self, video_path: str, sample_rate: int = 22050) -> Optional[str]:
        """Get cached audio or None if not cached.

        Args:
            video_path: Path to source video
            sample_rate: Audio sample rate

        Returns:
            Path to cached audio file, or None if not cached
        """
        file_hash = self._get_file_hash(video_path)
        cache_key = f"audio_{file_hash}_{sample_rate}"

        entry = self.index.get(cache_key)
        if entry and not entry.is_expired():
            audio_file = self.audio_dir / f"{cache_key}.wav"
            if audio_file.exists():
                return str(audio_file)

        return None

    def put_audio(self, video_path: str, audio_path: str, sample_rate: int = 22050) -> str:
        """Cache extracted audio file.

        Args:
            video_path: Path to source video
            audio_path: Path to extracted audio WAV file
            sample_rate: Audio sample rate

        Returns:
            Path where audio was cached
        """
        file_hash = self._get_file_hash(video_path)
        cache_key = f"audio_{file_hash}_{sample_rate}"
        cache_path = self.audio_dir / f"{cache_key}.wav"

        # Copy audio to cache
        with open(audio_path, "rb") as src, open(cache_path, "wb") as dst:
            dst.write(src.read())

        # Update index
        now = datetime.now()
        self.index[cache_key] = CacheEntry(
            hash=file_hash,
            source_path=str(video_path),
            source_size=Path(video_path).stat().st_size,
            source_mtime=Path(video_path).stat().st_mtime,
            cache_type="audio",
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=7)).isoformat(),
        )
        self._save_index()

        return str(cache_path)

    def get_proxy(self, video_path: str, height: int = 480) -> Optional[str]:
        """Get cached proxy video or None if not cached.

        Args:
            video_path: Path to source video
            height: Proxy height in pixels

        Returns:
            Path to cached proxy file, or None if not cached
        """
        file_hash = self._get_file_hash(video_path)
        cache_key = f"proxy_{file_hash}_{height}p"

        entry = self.index.get(cache_key)
        if entry and not entry.is_expired():
            proxy_file = self.proxy_dir / f"{cache_key}.mp4"
            if proxy_file.exists():
                return str(proxy_file)

        return None

    def put_proxy(self, video_path: str, proxy_path: str, height: int = 480) -> str:
        """Cache generated proxy video.

        Args:
            video_path: Path to source video
            proxy_path: Path to generated proxy MP4 file
            height: Proxy height in pixels

        Returns:
            Path where proxy was cached
        """
        file_hash = self._get_file_hash(video_path)
        cache_key = f"proxy_{file_hash}_{height}p"
        cache_path = self.proxy_dir / f"{cache_key}.mp4"

        # Copy proxy to cache
        with open(proxy_path, "rb") as src, open(cache_path, "wb") as dst:
            dst.write(src.read())

        # Update index
        now = datetime.now()
        self.index[cache_key] = CacheEntry(
            hash=file_hash,
            source_path=str(video_path),
            source_size=Path(video_path).stat().st_size,
            source_mtime=Path(video_path).stat().st_mtime,
            cache_type="proxy",
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=7)).isoformat(),
        )
        self._save_index()

        return str(cache_path)

    def get_metadata(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get cached detection metadata or None if not cached.

        Args:
            video_path: Path to source video

        Returns:
            Cached metadata dict, or None if not cached
        """
        file_hash = self._get_file_hash(video_path)
        cache_key = f"metadata_{file_hash}"

        entry = self.index.get(cache_key)
        if entry and not entry.is_expired():
            metadata_file = self.metadata_dir / f"{cache_key}.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        return json.load(f)
                except Exception:
                    return None

        return None

    def put_metadata(self, video_path: str, metadata: Dict[str, Any]) -> None:
        """Cache detection metadata.

        Args:
            video_path: Path to source video
            metadata: Detection results to cache
        """
        file_hash = self._get_file_hash(video_path)
        cache_key = f"metadata_{file_hash}"
        cache_path = self.metadata_dir / f"{cache_key}.json"

        with open(cache_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update index
        now = datetime.now()
        self.index[cache_key] = CacheEntry(
            hash=file_hash,
            source_path=str(video_path),
            source_size=Path(video_path).stat().st_size,
            source_mtime=Path(video_path).stat().st_mtime,
            cache_type="metadata",
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=7)).isoformat(),
        )
        self._save_index()

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        expired_keys = [
            k for k, v in self.index.items() if v.is_expired()
        ]

        for key in expired_keys:
            entry = self.index[key]

            # Remove cache file
            if entry.cache_type == "audio":
                cache_file = self.audio_dir / f"{key}.wav"
            elif entry.cache_type == "proxy":
                cache_file = self.proxy_dir / f"{key}.mp4"
            elif entry.cache_type == "metadata":
                cache_file = self.metadata_dir / f"{key}.json"
            else:
                continue

            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    pass

            del self.index[key]

        if expired_keys:
            self._save_index()

        return len(expired_keys)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, entry count, etc.)
        """
        total_size = 0
        count_by_type = {"audio": 0, "proxy": 0, "metadata": 0}

        for entry in self.index.values():
            if entry.cache_type == "audio":
                f = self.audio_dir / f"{entry.hash}_*.wav"
            elif entry.cache_type == "proxy":
                f = self.proxy_dir / f"{entry.hash}_*.mp4"
            elif entry.cache_type == "metadata":
                f = self.metadata_dir / f"{entry.hash}_*.json"
            else:
                continue

            count_by_type[entry.cache_type] += 1

        # Calculate total size
        for d in [self.audio_dir, self.proxy_dir, self.metadata_dir]:
            if d.exists():
                for f in d.glob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "entry_count": len(self.index),
            "by_type": count_by_type,
            "cache_dir": str(self.cache_dir),
        }
