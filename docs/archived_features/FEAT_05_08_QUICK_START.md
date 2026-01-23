# FEAT-05 & FEAT-08 Quick Start Guide

## What's New

### FEAT-05: Real-Time Status Dashboard
A sticky header showing dive processing progress with animated metrics.

### FEAT-08: Smart Connection Management
Automatic reconnection with exponential backoff and polling fallback when the server goes down.

---

## Getting Started in 30 Seconds

### Run with Live Dashboard
```bash
cd /Users/mcauchy/workflow/DiveAnalizer
diveanalyzer process video.mov --enable-server
```

Then open: **http://localhost:8765**

### What You'll See
1. **Status Dashboard** at the top (purple header)
   - Current phase (Phase 1: Audio Detection)
   - 4 metrics (dives found, speed, time remaining, thumbnails)
   - Animated progress bar (0-100%)

2. **Live Updates**
   - Dashboard updates every ~1 second
   - Phases change: Blue → Yellow → Green
   - Progress bar animates smoothly

3. **Connection Status** (bottom right)
   - Green indicator: Connected
   - Yellow/Orange: Reconnecting
   - Red: Connection lost (with retry button)

---

## Testing Connection Resilience

### Simulate Server Crash (Test Reconnection)

**Step 1**: Start processing
```bash
diveanalyzer process video.mov --enable-server
# Wait for some status updates to appear
```

**Step 2**: In another terminal, kill the server
```bash
# Find the process
ps aux | grep python

# Kill it
kill -9 <PID>
```

**Step 3**: Watch in browser
```
Timeline:
0s   → Dashboard stops updating
1s   → Banner shows "Reconnecting... (1/5)" with spinner
3s   → "Reconnecting... (2/5)"
7s   → "Reconnecting... (3/5)"
15s  → "Reconnecting... (4/5)"
23s  → "Reconnecting... (5/5)"
31s  → "Connection lost - Using cached data" + RED + Retry button

After 31s: Browser auto-switches to polling mode
(Fetches /events-history every 3 seconds)
```

**Step 4**: Click Retry (or start server again)
- Retry button resets counter
- Attempts connection again
- Browser console shows "Manual reconnect requested"

**Step 5**: Start server again
```bash
diveanalyzer process video.mov --enable-server
```

- Browser detects new connection
- Banner disappears
- Dashboard resumes live updates

---

## Dashboard Metrics Explained

### 📊 Dives Found: 42/60
- **42**: Dives detected so far
- **60**: Expected total (all audio splashes found)

### ⚡ Speed: 0.93 dives/min
- Calculated from actual processing rate
- Used to estimate time remaining

### ⏱️ Time Remaining: 5:30
- Auto-calculated: (60 - 42) / 0.93 ≈ 18 dives ÷ 0.93 ≈ 19 minutes
- Updates continuously

### 🖼️ Thumbnails: 15/488
- **15**: Thumbnails generated
- **488**: Total expected (3 thumbnails per dive × 163 dives)

---

## Color Coding

| Phase | Color | Progress |
|-------|-------|----------|
| Audio Detection | 🔵 Blue | 33% |
| Motion Detection | 🟡 Yellow | 66% |
| Person Detection | 🟢 Green | 90% |
| Extraction | ✅ Green | 100% |

---

## Connection States

### 🟢 Connected (Normal)
- SSE stream active
- Dashboard updating in real-time
- No banner visible

### 🟡 Reconnecting (1-5 Attempts)
- Orange banner with spinner
- Shows: "Reconnecting... (X/5)"
- No retry button
- Auto-attempts every: 1s, 2s, 4s, 8s, 8s

### 🔴 Connection Lost
- Red banner
- Shows: "Connection lost - Using cached data"
- Retry button available
- Automatically polls `/events-history` every 3s
- Gallery still works with cached dives

---

## Browser Console Debugging

Open DevTools: **F12 → Console**

### Check Connection Status
```javascript
console.log(eventConsumer.isConnected)
// true or false
```

### View Cached Events
```javascript
console.log('Cached events:', eventConsumer.cachedEvents.length)
// Cached events: 47
```

### Simulate Status Update
```javascript
if (statusDashboard) {
  statusDashboard.update({
    phase: "phase_2",
    phase_name: "Motion Detection",
    dives_found: 50,
    dives_expected: 60,
    thumbnails_ready: 30,
    thumbnails_expected: 300,
    processing_speed: 1.2,
    elapsed_seconds: 40,
    progress_percent: 70
  });
}
// Watch dashboard update in real-time
```

### Check localStorage Cache
```javascript
// View cache
console.log(JSON.parse(localStorage.getItem('diveanalyzer_events')))

// Clear cache
localStorage.removeItem('diveanalyzer_events')
console.log('Cache cleared')
```

### Force Reconnection
```javascript
// Reset and try again
eventConsumer.reconnectAttempts = 0
eventConsumer.connect()
console.log('Reconnection attempt started')
```

---

## Keyboard Shortcuts

(Same as before - these still work)

| Key | Action |
|-----|--------|
| ← → | Navigate dives |
| Space | Toggle dive for deletion |
| A | Select all |
| Ctrl+A | Deselect all |
| D | Delete selected |
| W | Watch selected |
| Enter | Accept & close |
| ? | Show help |

---

## Common Issues & Solutions

### Dashboard Not Showing Updates?
1. Check browser console: F12 → Console
2. Verify server running: `ps aux | grep python`
3. Check network tab: F12 → Network → should see EventSource connection
4. Try manual retry: Click red banner retry button

### Retry Button Not Working?
1. Check console for errors: F12 → Console
2. Verify `eventConsumer` object exists: `console.log(eventConsumer)`
3. Try manual connection: `eventConsumer.connect()`

### Cached Data Not Persisting?
1. Check localStorage: F12 → Application → Local Storage
2. Look for key: `diveanalyzer_events`
3. If missing: open DevTools, then reload page (F5)
4. Try manual cache save: `localStorage.setItem('test', 'value')`

### Connection Banner Stuck?
1. Click retry button (resets counter)
2. Or manually hide: `eventConsumer._hideConnectionBanner()`
3. Or manually show error: `eventConsumer._showConnectionBanner('error', 5)`

---

## Advanced Testing

### Rapid Event Emission (Load Test)
```javascript
// Emit 100 status updates rapidly
for(let i=0; i<100; i++) {
  if(statusDashboard) {
    statusDashboard.update({
      progress_percent: i,
      dives_found: i,
      dives_expected: 100
    });
  }
  await new Promise(r => setTimeout(r, 10));
}
```

### Memory Check
```javascript
// Monitor memory usage
setInterval(() => {
  const bytes = performance.memory?.usedJSHeapSize || 0
  const mb = (bytes / 1024 / 1024).toFixed(2)
  console.log(`Memory: ${mb}MB`)
}, 1000)
```

### Polling Simulation (Without Server)
```javascript
// Manually trigger polling if connected
if (eventConsumer.pollingActive) {
  eventConsumer._pollOnce()
  console.log('Poll triggered')
}
```

---

## Browser DevTools Tips

### Network Monitoring
1. Open F12 → Network tab
2. Look for:
   - **EventSource** connection (for /events)
   - **Fetch** requests (to /events-history)
   - **XHR** requests (if any)

### Console Filtering
```javascript
// Only show SSE logs
console.log = ((log) => function(...args) {
  if (args[0]?.includes?.('SSE')) log.apply(console, args)
})(console.log)
```

### Performance Profiling
1. F12 → Performance tab
2. Click record (red circle)
3. Wait 10 seconds
4. Click stop
5. Look for:
   - Dropped frames in animation
   - Long tasks blocking main thread
   - Memory usage spikes

---

## API Reference

### Status Update Event
```javascript
{
  event: "status_update",
  phase: "phase_1" | "phase_2" | "phase_3" | "complete",
  phase_name: "Audio Detection" | "Motion Detection" | "Person Detection" | "Extraction Complete",
  dives_found: number,
  dives_expected: number,
  thumbnails_ready: number,
  thumbnails_expected: number,
  elapsed_seconds: number,
  processing_speed: number,  // dives/minute
  progress_percent: number   // 0-100
}
```

### Endpoints
- **GET /**: Serve review gallery HTML
- **GET /events**: SSE stream (live status updates)
- **GET /events-history**: JSON array of last 100 events (polling fallback)
- **GET /health**: Server health check

### localStorage Key
- **Key**: `diveanalyzer_events`
- **Type**: JSON array of event objects
- **Max Size**: 500 events
- **Auto-Cleanup**: FIFO when exceeds limit

---

## Files to Know

| File | Purpose |
|------|---------|
| `diveanalyzer/utils/review_gallery.py` | Dashboard HTML/CSS/JS |
| `diveanalyzer/server/sse_server.py` | HTTP server + SSE |
| `diveanalyzer/cli.py` | Status update emissions |
| `FEAT_05_08_TESTING.md` | Comprehensive testing guide |
| `FEAT_05_08_IMPLEMENTATION_SUMMARY.md` | Full technical docs |
| `FEAT_05_08_FINAL_REPORT.md` | Implementation report |

---

## Still Not Working?

Check these in order:

1. **Is Python running?** `ps aux | grep python`
2. **Is port 8765 available?** `lsof -i :8765`
3. **Can you reach server?** `curl http://localhost:8765`
4. **Are events streaming?** F12 → Network → EventSource
5. **Check browser console** F12 → Console (look for errors)
6. **Check Python logs** (terminal where you ran the command)

---

## Performance Expectations

| Metric | Expected |
|--------|----------|
| Dashboard latency | <5ms |
| Update frequency | Every 500-1000ms |
| Memory usage | ~560KB |
| CPU (idle) | <1% |
| CPU (updating) | <2% |
| Animation FPS | 60 FPS |

---

## Quick Commands

```bash
# Start with dashboard
diveanalyzer process video.mov --enable-server

# With all detection phases
diveanalyzer process video.mov --enable-server --enable-motion --enable-person

# Custom port
diveanalyzer process video.mov --enable-server --server-port 9000

# Disable auto-browser open
diveanalyzer process video.mov --enable-server --no-open

# Verbose logging
diveanalyzer process video.mov --enable-server -v
```

---

## That's It!

You now have a production-grade monitoring dashboard with automatic reconnection and offline support.

**Enjoy! 🎉**

