"""Automatic cache cleanup for DiveAnalyzer.

Removes expired cache entries (older than 7 days) to save disk space.
Can be called manually or scheduled via cron job.
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
from .cache import CacheManager, get_cache_dir


def cleanup_expired_cache(cache_dir: Path = None, dry_run: bool = False) -> Dict[str, Any]:
    """Clean up expired cache entries.

    Args:
        cache_dir: Cache directory (defaults to ~/.diveanalyzer/cache)
        dry_run: If True, only report what would be deleted without actually deleting

    Returns:
        Dictionary with cleanup statistics
    """
    cache_dir = cache_dir or get_cache_dir()
    manager = CacheManager(cache_dir)

    stats = {
        "expired_count": 0,
        "freed_size_mb": 0.0,
        "entries_before": len(manager.index),
    }

    if dry_run:
        # Just count expired entries
        expired_count = sum(1 for e in manager.index.values() if e.is_expired())
        stats["expired_count"] = expired_count
        return stats

    # Actually clean up
    stats["expired_count"] = manager.cleanup_expired()
    cache_stats = manager.get_cache_stats()
    stats["freed_size_mb"] = cache_stats["total_size_mb"]
    stats["entries_after"] = len(manager.index)

    return stats


def get_cleanup_schedule_info() -> str:
    """Get information about setting up automatic cleanup.

    Returns:
        String with instructions for setting up cron job
    """
    return """
# Setup automatic cache cleanup (optional)

# For macOS/Linux, add to crontab (crontab -e):
# Run cleanup daily at 2 AM
0 2 * * * python3 -c "from diveanalyzer.storage import cleanup_expired_cache; cleanup_expired_cache()"

# Or use Python script:
# 0 2 * * * python3 /path/to/cleanup_script.py

# Check what will be deleted (dry run):
python3 -c "from diveanalyzer.storage import cleanup_expired_cache; print(cleanup_expired_cache(dry_run=True))"

# Actually clean up:
python3 -c "from diveanalyzer.storage import cleanup_expired_cache; print(cleanup_expired_cache())"
"""
