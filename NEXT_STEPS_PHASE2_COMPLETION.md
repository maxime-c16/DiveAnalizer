# Next Steps: Phase 2 Completion (Proxy Integration)

## Current Bottleneck

**Motion detection is 150s on full video - unacceptable for production**

```
Current Phase 2 Performance:
├─ Audio extraction: 1.2s ✓ Fast
├─ Splash detection: 0.3s ✓ Fast
├─ Motion detection: 150s ✗ SLOW (full video)
├─ Fusion: 0.0s ✓ Fast
└─ Total: 150s ✗ Not practical

Expected with 480p proxy:
├─ Proxy generation: 60s (one-time, cached)
├─ Motion on proxy: 2-3s ✓ Fast!
└─ Total: 2-3s (or 65s first run, 2s cached) ✓ Production-ready
```

---

## What's Already Built

✅ Proxy generation code exists: `diveanalyzer/extraction/proxy.py`
✅ Cache system exists: `diveanalyzer/storage/cache.py`
✅ Motion detection works: `diveanalyzer/detection/motion.py`
✅ CLI has --enable-motion flag
❌ **NOT integrated**: Proxy isn't actually used by motion detection

---

## The CLEAR NEXT STEP

### Step 1: Integrate Proxy into Motion Detection Workflow

**File to modify**: `diveanalyzer/cli.py` (process command)

**Current code** (WRONG - uses full video):
```python
if enable_motion:
    motion_events = detect_motion_bursts(video_path, sample_fps=5.0)
```

**New code** (CORRECT - uses proxy):
```python
if enable_motion:
    # Generate or get cached proxy
    if is_proxy_generation_needed(video_path):
        click.echo("  Generating 480p proxy...")
        proxy_path = "/tmp/proxy.mp4"
        generate_proxy(video_path, proxy_path, height=proxy_height)
        motion_video = proxy_path
    else:
        motion_video = video_path

    # Run motion detection on proxy (2-3s instead of 150s!)
    click.echo("  Detecting motion...")
    motion_events = detect_motion_bursts(motion_video, sample_fps=5.0)
```

### Step 2: Update Person Detection (Phase 3 prep)

Same pattern: Generate proxy once, use for both motion + person detection

### Step 3: Add Caching Integration

Cache the proxy so we don't regenerate on next run:
```python
from diveanalyzer.storage import CacheManager

cache = CacheManager()
proxy_path = cache.get_proxy(video_path, height=480)

if not proxy_path:
    # Generate and cache
    tmp_proxy = "/tmp/proxy.mp4"
    generate_proxy(video_path, tmp_proxy, height=480)
    proxy_path = cache.put_proxy(video_path, tmp_proxy, height=480)
```

---

## Implementation Plan (30 minutes)

### 1. Create new helper function (5 min)

```python
# Add to diveanalyzer/cli.py

def get_or_generate_proxy(video_path, proxy_height=480, enable_cache=True):
    """Get or generate 480p proxy for motion/person detection."""
    from .storage import CacheManager

    # Check if proxy generation is needed
    if not is_proxy_generation_needed(video_path):
        return video_path  # Video is already small enough

    if enable_cache:
        cache = CacheManager()
        cached_proxy = cache.get_proxy(video_path, height=proxy_height)
        if cached_proxy:
            return cached_proxy

    # Generate new proxy
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        proxy_path = tmp.name

    generate_proxy(video_path, proxy_path, height=proxy_height, verbose=False)

    if enable_cache:
        cache = CacheManager()
        proxy_path = cache.put_proxy(video_path, proxy_path, height=proxy_height)

    return proxy_path
```

### 2. Update process command (10 min)

Replace this:
```python
if enable_motion:
    click.echo("\n🎬 Phase 2: Motion-Based Validation...")
    try:
        # ... existing code that uses full video
        motion_events = detect_motion_bursts(video_path, sample_fps=5.0)
```

With this:
```python
if enable_motion:
    click.echo("\n🎬 Phase 2: Motion-Based Validation...")
    try:
        # Get or generate proxy
        if verbose:
            click.echo("  Getting proxy for motion detection...")
        proxy_path = get_or_generate_proxy(
            video_path,
            proxy_height=proxy_height,
            enable_cache=enable_cache
        )

        click.echo("  Detecting motion bursts...")
        motion_events = detect_motion_bursts(proxy_path, sample_fps=5.0)
```

### 3. Test on real video (10 min)

```bash
# Test Phase 2 with proxy
time python3 -m diveanalyzer process IMG_6496.MOV \
    -o /tmp/dives_phase2_proxy \
    --enable-motion \
    --verbose

# Expected timing:
# - First run: ~65s (60s proxy + 2-3s motion + rest)
# - Cached run: ~2-3s (proxy cached, just motion + extraction)
```

### 4. Verify results (5 min)

```bash
# Check cache
ls -lh ~/.diveanalyzer/cache/

# Verify same dives detected
ls /tmp/dives_phase2_proxy/dive_*.mp4 | wc -l
# Should be 30 dives
```

---

## Expected Results

### Before (Full video motion detection):
```
🎬 Phase 2: Motion-Based Validation...
  Detecting motion on full video...
  ✓ Found 47 motion bursts
  ⏱️  Motion detection: 150s ✗
```

### After (Proxy motion detection):
```
🎬 Phase 2: Motion-Based Validation...
  Getting proxy for motion detection...
  Generating 480p proxy...
  ✓ Proxy generated: /home/.../proxy_IMG_6496.mp4 (50MB)
  Detecting motion on proxy...
  ✓ Found 47 motion bursts
  ⏱️  Motion detection: 2.5s ✓
```

---

## Why This is the Clear Next Step

1. **Highest impact**: 150s → 2-3s (100x speedup!)
2. **Unblocks Phase 3**: Person detection also needs proxy
3. **Simple**: Just pass proxy_path instead of video_path
4. **Proven code**: Proxy generation and cache already work
5. **Immediate value**: Makes Phase 2 production-ready

---

## After This

Once proxy integration works:

1. **Phase 2 is COMPLETE** ✅
   - Audio detection: 0.3s
   - Motion detection: 2-3s (on proxy)
   - Total: 2-3s (first run) or <1s (cached)
   - Confidence: 0.92 average

2. **Ready for Phase 3**
   - Person detection uses same proxy
   - Estimated 3s for YOLO inference
   - Total: 5-6s for all three signals

3. **Production deployment**
   - Fast enough for batch processing
   - Cache system enables reuse
   - CLI is complete with all features

---

## Code Changes Summary

| File | Change | Lines | Time |
|------|--------|-------|------|
| `cli.py` | Add `get_or_generate_proxy()` function | 20 | 5 min |
| `cli.py` | Update `process` command motion section | 10 | 5 min |
| Testing | Run on real video, verify timing | - | 10 min |
| Commit | Create new commit with proxy integration | - | 5 min |

**Total: ~30 minutes**

---

## Success Criteria

✅ Motion detection runs on 480p proxy (not full video)
✅ Performance improves to <3s for motion detection
✅ Same 47 motion bursts detected as before
✅ Cache stores proxy for reuse
✅ CLI shows improvement in timing

---

## Then What?

After proxy integration works:

1. **Phase 3**: Implement person detection on same proxy (3 weeks)
2. **Optimization**: GPU acceleration for motion+person (optional)
3. **Production**: Deploy with --enable-motion flag as default
4. **Advanced**: Add adaptive thresholds, auto-zone calibration

---

## Ready to Implement?

This is clearly the next step because:
- ✅ All dependencies exist
- ✅ Well-defined change (add proxy to motion pipeline)
- ✅ High impact (100x speedup)
- ✅ No architectural changes needed
- ✅ Can test immediately on real video
- ✅ Unblocks everything else (Phase 3, deployment)

**Shall I implement this now?**
