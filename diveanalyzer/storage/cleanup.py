"""Automatic cache cleanup for DiveAnalyzer.

Removes expired cache entries (older than 7 days) to save disk space.
Can be called manually, scheduled via cron job, or run as background daemon.
"""

import shutil
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from .cache import CacheManager, get_cache_dir


def cleanup_expired_cache(cache_dir: Path = None, dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """Clean up expired cache entries.

    Args:
        cache_dir: Cache directory (defaults to ~/.diveanalyzer/cache)
        dry_run: If True, only report what would be deleted without actually deleting
        verbose: Print detailed information

    Returns:
        Dictionary with cleanup statistics
    """
    cache_dir = cache_dir or get_cache_dir()
    manager = CacheManager(cache_dir)

    # Get stats before cleanup
    stats_before = manager.get_cache_stats()

    stats = {
        "expired_count": 0,
        "freed_size_mb": 0.0,
        "entries_before": len(manager.index),
        "entries_after": len(manager.index),
        "total_size_before_mb": stats_before["total_size_mb"],
        "total_size_after_mb": stats_before["total_size_mb"],
        "timestamp": datetime.now().isoformat(),
    }

    if dry_run:
        # Count expired entries without deleting
        expired = [(k, e) for k, e in manager.index.items() if e.is_expired()]
        stats["expired_count"] = len(expired)

        if verbose and expired:
            print(f"Would delete {len(expired)} expired entries:")
            for hash_key, entry in expired:
                age_days = (datetime.now() - datetime.fromisoformat(entry.created_at)).days
                print(f"  • {Path(entry.source_path).name} ({age_days} days old)")

        return stats

    # Actually clean up
    deleted_count = manager.cleanup_expired()
    stats_after = manager.get_cache_stats()

    stats["expired_count"] = deleted_count
    stats["entries_after"] = len(manager.index)
    stats["freed_size_mb"] = max(0, stats_before["total_size_mb"] - stats_after["total_size_mb"])
    stats["total_size_after_mb"] = stats_after["total_size_mb"]

    if verbose and deleted_count > 0:
        print(f"✓ Cleaned up {deleted_count} expired entries")
        print(f"  Freed: {stats['freed_size_mb']:.1f} MB")
        print(f"  Cache size: {stats['total_size_after_mb']:.1f} MB → {stats_after['total_size_mb']:.1f} MB")

    return stats


def get_cache_stats(cache_dir: Path = None, detailed: bool = False) -> Dict[str, Any]:
    """Get detailed cache statistics.

    Args:
        cache_dir: Cache directory (defaults to ~/.diveanalyzer/cache)
        detailed: Include breakdown by type (audio/proxy/metadata)

    Returns:
        Dictionary with cache statistics
    """
    cache_dir = cache_dir or get_cache_dir()
    manager = CacheManager(cache_dir)
    stats = manager.get_cache_stats()

    # Add expiration info
    total_entries = len(manager.index)
    expired_entries = sum(1 for e in manager.index.values() if e.is_expired())
    valid_entries = total_entries - expired_entries

    stats.update({
        "total_entries": total_entries,
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
    })

    if detailed:
        # Breakdown by type
        by_type = {}
        for entry in manager.index.values():
            type_name = entry.cache_type
            if type_name not in by_type:
                by_type[type_name] = {"count": 0, "size_mb": 0}
            by_type[type_name]["count"] += 1
            # Approximate size calculation
            if entry.cache_type == "proxy":
                by_type[type_name]["size_mb"] += 50  # Typical proxy is ~50MB
            elif entry.cache_type == "audio":
                by_type[type_name]["size_mb"] += 5  # Typical audio is ~5MB
            else:
                by_type[type_name]["size_mb"] += 0.1  # Metadata is tiny

        stats["by_type"] = by_type

    return stats


def check_disk_space(cache_dir: Path = None, warn_threshold_mb: int = 500) -> Dict[str, Any]:
    """Check available disk space.

    Args:
        cache_dir: Cache directory (defaults to ~/.diveanalyzer/cache)
        warn_threshold_mb: Warn if free space below this (MB)

    Returns:
        Dictionary with disk space information
    """
    cache_dir = cache_dir or get_cache_dir()

    # Get disk usage
    disk_usage = shutil.disk_usage(cache_dir)

    cache_stats = get_cache_stats(cache_dir)

    info = {
        "total_gb": disk_usage.total / (1024**3),
        "used_gb": disk_usage.used / (1024**3),
        "free_gb": disk_usage.free / (1024**3),
        "cache_size_mb": cache_stats["total_size_mb"],
        "percent_full": (disk_usage.used / disk_usage.total) * 100,
        "status": "OK",
    }

    # Check if we should warn
    if disk_usage.free / (1024**2) < warn_threshold_mb:
        info["status"] = "LOW_SPACE"
        info["warning"] = f"Free disk space is only {info['free_gb']:.1f} GB"

    return info


class CacheCleanupScheduler:
    """Background scheduler for automatic cache cleanup."""

    def __init__(self, cache_dir: Path = None, interval_hours: int = 24, auto_start: bool = False):
        """Initialize scheduler.

        Args:
            cache_dir: Cache directory
            interval_hours: How often to run cleanup (default: 24 hours)
            auto_start: Start scheduler immediately
        """
        self.cache_dir = cache_dir or get_cache_dir()
        self.interval_hours = interval_hours
        self.interval_seconds = interval_hours * 3600
        self.running = False
        self.thread = None
        self.stats = []

        if auto_start:
            self.start()

    def start(self) -> None:
        """Start the background cleanup scheduler."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the background cleanup scheduler."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run_loop(self) -> None:
        """Main scheduler loop (runs in background thread)."""
        while self.running:
            try:
                # Run cleanup
                stats = cleanup_expired_cache(cache_dir=self.cache_dir, verbose=False)
                self.stats.append(stats)

                # Keep only last 30 cleanup records
                if len(self.stats) > 30:
                    self.stats = self.stats[-30:]

            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")

            # Sleep until next interval
            time.sleep(self.interval_seconds)

    def get_last_run(self) -> Optional[Dict[str, Any]]:
        """Get statistics from last cleanup run."""
        return self.stats[-1] if self.stats else None


def get_cleanup_schedule_info() -> str:
    """Get information about setting up automatic cleanup.

    Returns:
        String with instructions for setting up cron job
    """
    return """
# Setup automatic cache cleanup (optional)

# For macOS/Linux, add to crontab (crontab -e):
# Run cleanup daily at 2 AM
0 2 * * * python3 -c "from diveanalyzer.storage.cleanup import cleanup_expired_cache; cleanup_expired_cache()"

# Or in Python code:
# from diveanalyzer.storage.cleanup import CacheCleanupScheduler
# scheduler = CacheCleanupScheduler(auto_start=True)  # Starts 24-hour cleanup loop

# Check what will be deleted (dry run):
# python3 -c "from diveanalyzer.storage.cleanup import cleanup_expired_cache; print(cleanup_expired_cache(dry_run=True, verbose=True))"

# Actually clean up:
# python3 -c "from diveanalyzer.storage.cleanup import cleanup_expired_cache; print(cleanup_expired_cache())"

# Get cache stats:
# python3 -c "from diveanalyzer.storage.cleanup import get_cache_stats; print(get_cache_stats(detailed=True))"

# Check disk space:
# python3 -c "from diveanalyzer.storage.cleanup import check_disk_space; print(check_disk_space())"
"""
