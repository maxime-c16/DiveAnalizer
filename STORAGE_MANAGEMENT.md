# Storage Management Guide

**Status**: Keep storage tight with smart caching and cleanup

---

## Disk Space Usage

### Typical Processing

Per video session (8-10 min, 500MB+):

```
INPUT:        500MB  (original video)
CACHE:         50MB  (480p proxy - reused)
OUTPUT:       200MB  (30 dive clips × 6-7MB each)
TEMP:         ~0MB   (cleaned automatically)
────────────────────
TOTAL:        750MB per session
```

### Multi-Video Processing

5 sessions:
- Without caching: 2.5GB (each session keeps temp files)
- **With caching**: 600MB (proxies reused, single copy)
- **Savings**: 2.0GB (75% reduction!)

---

## Cache Management

### Cache Location
```
~/.diveanalyzer/cache/
├── audio/       (Audio extracts - cleaned after 7 days)
├── proxies/     (480p proxies - reused across sessions)
└── metadata/    (Index for quick lookup)
```

### Auto-Cleanup

The system automatically removes cache entries older than **7 days**:

```python
# Entries expire after 7 days
# Run this periodically to clean:
diveanalyzer clear-cache

# Or cron job for automatic cleanup
0 2 * * * diveanalyzer clear-cache 2>/dev/null
```

---

## Storage Optimization Tips

### 1. Check Cache Status Before Running

```bash
# See current cache usage
diveanalyzer clear-cache

# Output:
# Cache directory: ~/.diveanalyzer/cache/
# Total cached items: 5
# Total cache size: 245.3 MB
```

### 2. Manual Cleanup When Needed

```bash
# Dry-run: See what would be deleted
diveanalyzer clear-cache --dry-run

# Would delete:
# - Expired audio (older than 7 days): 120MB
# - Expired metadata: 5MB

# Actually delete
diveanalyzer clear-cache
```

### 3. Clean Old Dive Outputs

```bash
# Remove extracted clips older than N days
find ~/Dives -type f -mtime +30 -delete

# Keep only recent extractions
find ~/Dives -name "*.mp4" -type f -exec ls -lh {} \; | sort -k6 | head -20
```

### 4. Archive or Delete Large Videos

```bash
# Move old sessions to external drive
mv Session_*.mov /Volumes/ExternalDrive/Archive/

# Or delete if no longer needed
rm IMG_OLD_*.MOV
```

---

## Best Practices

### During Development/Testing

```bash
# Test processing (don't extract clips yet)
diveanalyzer detect IMG_6496.MOV

# Just analyze (minimal storage)
diveanalyzer analyze-motion IMG_6496.MOV
diveanalyzer analyze-audio IMG_6496.MOV
```

### Before Processing Large Batches

```bash
# Clean up old cache
diveanalyzer clear-cache

# Check available disk space
df -h

# Ensure at least 2GB free
```

### After Processing Sessions

```bash
# Keep only essential clips
# Archive old extractions
ls -1d dives_* | head -n -3 | xargs -I {} tar czf {}.tar.gz {} && rm -rf {}

# Remove temp directories
rm -rf /tmp/diveanalyzer_*
rm -rf /tmp/proxy_*
```

---

## Storage Budget

### Recommended Disk Space

For comfortable operation:

```
Video library:      ~500MB
Active proxies:     ~100MB
Recent clips:       ~500MB
Working temp space: ~500MB
─────────────────────────
TOTAL:            ~1.6GB minimum

Comfortable:      ~3-5GB
Heavy use:        ~10GB+
```

### Cleanup Triggers

Automatically clean when:
- Total cache > 500MB
- Available disk < 1GB
- Cache entries > 100

---

## Automated Cleanup

### Option 1: Weekly Cron Job

```bash
# Edit crontab
crontab -e

# Add this line (runs every Monday at 2 AM):
0 2 * * 1 diveanalyzer clear-cache 2>/dev/null

# Verify
crontab -l
```

### Option 2: Manual Reminder

```bash
# Check cache status weekly
alias check-cache='diveanalyzer clear-cache'

# Add to ~/.zshrc or ~/.bash_profile
echo "alias check-cache='diveanalyzer clear-cache'" >> ~/.zshrc
```

### Option 3: Script-Based Cleanup

```bash
#!/bin/bash
# cleanup.sh - Clean cache and old clips

echo "🧹 Cleaning DiveAnalyzer cache..."
diveanalyzer clear-cache

echo "📁 Archiving old clips..."
cd ~/Dives 2>/dev/null || exit 0
tar czf archive_$(date +%Y%m%d).tar.gz dive_*.mp4 && rm dive_*.mp4
ls -lh archive_*.tar.gz | head -5

echo "✓ Cleanup complete"
```

---

## Storage Compression

### iPhone Video Format

The iOS videos are already quite efficient:

```
IMG_6496.MOV (8 min, 520MB)
├─ Video codec: H.264 (efficient)
├─ Resolution: 1920×1080 @ 30fps
└─ Bitrate: ~10 Mbps

480p proxy:
├─ Resolution: 854×480
├─ Bitrate: ~2-3 Mbps
└─ Size: 33MB (94% reduction)
```

### Extraction Optimization

Output clips are already optimized:

```
dive_001.mp4 (6 seconds)
├─ Stream copy (no re-encoding)
├─ Size: ~6MB
└─ Quality: Same as original

30 dives: ~180MB total
```

---

## Monitoring

### Check Storage Weekly

```bash
# Quick status
du -sh ~/.diveanalyzer/cache
du -sh ~/Dives
df -h

# Detailed cache info
diveanalyzer clear-cache
```

### Storage Alerts

Create a status script:

```bash
#!/bin/bash
# storage_alert.sh

cache_size=$(du -sh ~/.diveanalyzer/cache 2>/dev/null | cut -f1)
free_space=$(df -h . | tail -1 | awk '{print $4}')

echo "Cache: $cache_size"
echo "Free space: $free_space"

if [[ "$free_space" < "1G" ]]; then
  echo "⚠️  LOW DISK SPACE!"
  diveanalyzer clear-cache
fi
```

---

## FAQ

### Q: How much space do proxies take?
**A**: 480p proxies are typically 50-100MB each. With caching, you keep only 1 copy per unique video, not one per processing run.

### Q: Should I delete proxies manually?
**A**: No, let the cache system manage them. They'll auto-expire after 7 days. Manual deletion is only if you need space urgently.

### Q: Can I reduce proxy size further?
**A**: Yes, use lower resolution (360p, 240p) via `--proxy-height 360` but accuracy may suffer. 480p is recommended.

### Q: How often should I clean cache?
**A**: Automatically (7-day auto-cleanup) or manually when available space < 2GB.

### Q: What takes the most space?
**A**:
1. Output clips (~180MB per session)
2. Original videos (~500MB each)
3. Proxies (~50MB each, but reused)

### Q: Can I archive clips to cloud?
**A**: Yes, but processing requires local video. Archive extracted clips:
```bash
diveanalyzer process video.mov && \
tar czf clips.tar.gz dives/ && \
rm -rf dives/
```

---

## Cleanup Checklist

### Weekly
- [ ] Check: `diveanalyzer clear-cache`
- [ ] Verify available disk space: `df -h`
- [ ] Archive old clips if > 500MB

### Monthly
- [ ] Delete old videos if retained locally
- [ ] Review cache effectiveness
- [ ] Check storage trend

### As Needed
- [ ] When disk < 2GB: Manual cleanup
- [ ] After large batch: Archive outputs
- [ ] Before new session: Verify free space

---

## Example: Tight Storage Scenario

**Scenario**: iPhone + iCloud, only 256GB storage

```bash
# Keep videos only on iCloud
iCloud Photos: Videos (~500MB each)

# Process to local temp
diveanalyzer process /path/to/icloud/video.mov --enable-motion

# Output clips directly to iCloud
mkdir ~/Library/Mobile\ Documents/com~apple~CloudDocs/DiveClips/
mv dives/* ~/Library/Mobile\ Documents/com~apple~CloudDocs/DiveClips/

# Cache is small and local
du -sh ~/.diveanalyzer/cache  # Usually < 200MB

# Local temp: ~100MB (cleaned after)
```

---

## Summary

✅ **Smart caching keeps storage efficient**
- Proxy reuse: 94% space savings
- Auto-cleanup: 7-day expiration
- Small temp footprint: ~0 after processing

📊 **Monitor regularly**
- Weekly cache check
- Archive old clips
- Keep 2GB free space

🧹 **Simple cleanup**
- One command: `diveanalyzer clear-cache`
- Optional: Cron job for automation
- Manual: When needed for space

---

**Remember**: Clean cache regularly, archive important clips, and keep 2GB free for comfortable processing!
