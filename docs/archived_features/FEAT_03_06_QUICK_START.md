# FEAT-03 & FEAT-06 Quick Start Guide

## For CLI Users

### Basic Usage

```bash
# Enable server and auto-open browser (default)
diveanalyzer process video.mov --enable-server

# Disable auto-launch if needed
diveanalyzer process video.mov --enable-server --no-open

# Custom port
diveanalyzer process video.mov --enable-server --server-port 9000
```

### What You See

1. **Browser opens** showing empty gallery with "Waiting for dives..." message
2. **As dives detected**, placeholder cards appear with shimmer animation
3. **User can interact** immediately - no need to wait for extraction
4. **Placeholders replaced** with actual thumbnails as they become available

---

## For Backend Developers

### Emitting Dive Detected Events

To trigger placeholder rendering, emit `dive_detected` event:

```python
# In your detection code
server.emit("dive_detected", {
    "dive_index": 1,              # Required: dive number
    "dive_id": "dive_001",        # Optional: unique ID
    "duration": 1.25,             # Optional: seconds
    "confidence": 0.95,           # Optional: 0.0-1.0
})
```

### Event Timing

- Emit as soon as dive is **detected** (not extracted)
- Placeholder appears immediately on client
- Later: emit thumbnail data to replace placeholder

---

## For Frontend Developers

### Using renderDiveCardPlaceholder()

```javascript
// Manually create placeholder (for testing)
renderDiveCardPlaceholder({
  dive_index: 1,
  dive_id: "dive_001",
  duration: 1.2,
  confidence: 0.95
});
```

### CSS Classes Reference

**For styling custom elements:**
- `.placeholder-card` - Main container with fade-in
- `.placeholder-thumbnails` - 3-column thumbnail area
- `.placeholder-thumbnail` - Individual shimmer effect
- `.placeholder-info` - Bottom info section
- `.placeholder-number` - Dive number skeleton
- `.placeholder-detail` - Duration/status skeletons
- `.placeholder-confidence` - Confidence badge skeleton
- `.empty-gallery-message` - "Waiting for dives..." state

### Overriding Animations

```css
/* Faster shimmer */
.placeholder-thumbnail {
  animation: shimmer 1s infinite; /* default 2s */
}

/* Different fade-in speed */
.placeholder-card {
  animation: fadeIn 0.1s ease-in; /* default 0.2s */
}
```

---

## Testing

### Minimal Test (No Server)

1. Open `review_gallery.html` in browser
2. Open browser console
3. Run:
   ```javascript
   renderDiveCardPlaceholder({
     dive_index: 1,
     dive_id: "test_dive",
     duration: 1.5,
     confidence: 0.9
   });
   ```
4. Should see placeholder with shimmer

### Full Integration Test

```bash
# Terminal 1: Start processing
diveanalyzer process sample.mov --enable-server --verbose

# Terminal 2 (optional): Monitor events
# Watch browser console for dive_detected events
```

---

## Troubleshooting

### Browser Doesn't Open

**Check:**
1. Port 8765 is available: `lsof -i :8765`
2. Use `--no-open` to bypass and manually navigate
3. Check `--verbose` output for errors

### Placeholders Don't Appear

**Check:**
1. Browser console for errors
2. Server connection status in top-right
3. Network tab to verify SSE connection
4. Events tab to see incoming events

### Shimmer Animation Stutters

**Fix:**
1. Check system CPU load
2. Try disabling other browser tabs
3. Animation is GPU-accelerated (should be smooth)

---

## Configuration

### Via CLI Flags

- `--enable-server`: Start HTTP server
- `--server-port`: Custom port (default 8765)
- `--no-open`: Don't auto-launch browser
- `--verbose`: Show detailed event logs

### Browser Events

All events logged in Event Log (bottom-right icon):
- Connection status
- Dive detected count
- Processing stage
- Errors/warnings

---

## Next Steps

### For Thumbnails Enhancement

1. Detect when thumbnail ready
2. Emit event with thumbnail data URL
3. JavaScript replaces placeholder with actual image
4. Smooth fade transition already supports this

### For Progress Display

1. Emit multiple events as dive processes
2. Update placeholder with progress indicator
3. Show estimated completion time

---

## FAQ

**Q: Can I disable the browser launch?**  
A: Yes, use `--no-open` flag

**Q: What if my server is on different machine?**  
A: Modify server URL in browser console (advanced)

**Q: Do placeholders slow down page?**  
A: No - CSS animations are GPU-accelerated, minimal DOM operations

**Q: Can I style the placeholders?**  
A: Yes - all elements use CSS classes, easy to customize

**Q: What if server crashes?**  
A: Placeholders remain, shows "disconnected" status, graceful degradation

---

## Performance Tips

1. **Batch events** - Don't emit duplicate events
2. **Lazy thumbnails** - Only generate when needed
3. **Progressive loading** - Start with low-res, upgrade to high-res
4. **Monitor memory** - Each placeholder ~2KB, manageable even for 100+ dives

---

## Support

For issues or questions:
1. Check Event Log (browser console)
2. Verify SSE connection in browser DevTools
3. Run with `--verbose` for detailed CLI output
4. Check server logs if available
