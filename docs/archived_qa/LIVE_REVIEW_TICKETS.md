# Live Dive Review - UX Enhancement Tickets

**Objective**: User runs CLI → HTML opens immediately → sees dives appear in real-time → thumbnails populate as they generate

**Architecture**: HTTP Server + Server-Sent Events (SSE) + Live HTML Updates

---

## FEAT-01: HTTP Server & SSE Endpoint Integration
**Priority**: CRITICAL (Foundation for all other features)

Create lightweight HTTP server that serves HTML and provides real-time event stream.

### Requirements:
- Add HTTP server to CLI (Flask/FastAPI minimal wrapper)
- Implement `/events` SSE endpoint that streams dive detection and thumbnail progress
- Serve `index.html` at root URL
- Auto-start server when processing video
- Listen on `http://localhost:8765` (configurable)

### Acceptance Criteria:
- Server starts without blocking video processing
- SSE endpoint accepts connections and streams JSON events
- HTML can connect and receive real-time updates
- Server gracefully shuts down when processing completes

### Files to Modify:
- `diveanalyzer/cli.py` - Add server startup logic
- `diveanalyzer/server/sse_server.py` (NEW) - SSE server implementation

---

## FEAT-02: HTML Real-Time Event Consumer
**Priority**: CRITICAL (Consumes server events)

Update HTML to connect to SSE endpoint and process real-time dive/thumbnail updates.

### Requirements:
- JavaScript to connect to `/events` SSE stream
- Parse incoming JSON events (dive_detected, thumbnail_ready, status_update)
- Update DOM in real-time without full page reload
- Handle disconnection and auto-reconnect with exponential backoff
- Show connection status indicator

### Acceptance Criteria:
- HTML connects on page load
- Events are consumed and logged to console
- DOM updates are smooth (no flickering)
- Reconnection works after network interruption
- Status indicator shows "Connected" or "Disconnected"

### Files to Modify:
- `diveanalyzer/utils/review_gallery.py` - Update HTML template to include SSE consumer JavaScript

---

## FEAT-03: Dive Card Placeholder System
**Priority**: HIGH (Shows dives immediately)

Display dive cards with placeholders before thumbnails arrive.

### Requirements:
- When `dive_detected` event received: immediately render card with placeholder graphics
- Show dive number, duration, confidence (from detection data)
- Placeholder: light gray skeleton/shimmer loading state
- Layout adapts as cards are added (2-column grid)
- Preserve scroll position as new cards appear

### Acceptance Criteria:
- New dive appears in gallery within 100ms of detection event
- Card shows all available data (number, duration, confidence)
- Placeholder is visually distinct from loaded state
- No layout shift/jank as cards appear
- User can interact with modal even during placeholder state

### Files to Modify:
- `diveanalyzer/utils/review_gallery.py` - Add placeholder HTML template

---

## FEAT-04: Progressive Thumbnail Loading
**Priority**: HIGH (Core UX improvement)

Update thumbnails in-place as they're generated, without reloading page.

### Requirements:
- When `thumbnail_ready` event received: update specific dive card with new frames
- Thumbnail update is animated fade-in (200ms)
- 3-frame grid thumbnail updates first
- 8-frame timeline updates second (when modal is opened)
- Old placeholder fades out, new frames fade in smoothly
- Track which thumbnails are ready vs pending

### Acceptance Criteria:
- Thumbnail appears in gallery within 500ms of generation
- Fade animation is smooth and doesn't block interactions
- Multiple concurrent thumbnail updates work correctly
- User can click "ready" thumbnails immediately, see "loading" for pending
- No console errors on thumbnail update

### Files to Modify:
- `diveanalyzer/utils/review_gallery.py` - Add thumbnail update JavaScript

---

## FEAT-05: Status Dashboard & Progress Tracking
**Priority**: HIGH (Shows user what's happening)

Display real-time status of algorithm and CLI operations.

### Requirements:
- Header area showing:
  - Current processing phase (Phase 1/2/3, audio vs motion vs person)
  - Processing speed (dives/minute, frames/second)
  - Estimated time remaining
  - Dives detected so far (e.g., "42 of ~60 expected")
  - Thumbnail generation progress (e.g., "Building thumbnails: 15/42")
- Status messages like "🔍 Analyzing audio...", "🎬 Detecting motion...", "✅ Person detection enabled"
- Progress bar for overall completion

### Acceptance Criteria:
- Status updates at least every 500ms
- All metrics are human-readable
- Progress bar smoothly animates
- Colors indicate phase (audio=blue, motion=yellow, person=green)
- Status dashboard doesn't consume excessive bandwidth

### Files to Modify:
- `diveanalyzer/utils/review_gallery.py` - Add status dashboard HTML/CSS
- `diveanalyzer/server/sse_server.py` - Emit status events

---

## FEAT-06: Auto-Launch Browser
**Priority**: MEDIUM (Quality of life)

Automatically open HTML in default browser when CLI starts processing.

### Requirements:
- When HTTP server starts, automatically open `http://localhost:8765` in default browser
- Works on macOS, Linux, Windows
- User can disable with `--no-open` flag
- Only opens once (not on every event)
- Opens BEFORE video processing starts (so user sees empty gallery immediately)

### Acceptance Criteria:
- Default browser opens to gallery URL
- Works on all platforms
- Can be disabled with flag
- No errors if browser is unavailable
- Page loads before first dives are detected

### Files to Modify:
- `diveanalyzer/cli.py` - Add browser launch logic

---

## FEAT-07: Deferred Thumbnail Generation
**Priority**: CRITICAL (Enables fast start)

Generate thumbnails AFTER dives are detected so gallery appears quickly.

### Requirements:
- Timeline:
  1. Dives detected (Phase 1: audio) → send `dive_detected` event (no thumbnails yet)
  2. Gallery shows dives with placeholders immediately
  3. User can start reviewing (select, watch, delete)
  4. After all dives detected OR timeout: start thumbnail generation pipeline
  5. Each thumbnail ready → send `thumbnail_ready` event → HTML updates
- Thumbnail generation should NOT block dive detection
- Use background thread/process for thumbnail generation
- Can generate 3-frame gallery thumbnails and 8-frame timeline thumbnails separately

### Acceptance Criteria:
- Gallery visible within 2-3 seconds of starting analysis (before thumbnails)
- All dives detected before thumbnail generation starts
- Thumbnails appear progressively as they're ready
- User can interact with gallery while thumbnails generate
- No frame drops or jank during thumbnail generation

### Files to Modify:
- `diveanalyzer/cli.py` - Restructure dive detection and thumbnail generation
- `diveanalyzer/utils/review_gallery.py` - Add deferred thumbnail pipeline

---

## FEAT-08: Connection Management & Fallback
**Priority**: MEDIUM (Robustness)

Handle WebSocket disconnections and provide graceful fallback.

### Requirements:
- SSE reconnection with exponential backoff (1s → 2s → 4s → 8s max)
- If connection lost: show warning, disable live updates, show cached data
- If server crashes: detect and show "Lost connection" message
- Fallback to polling if SSE fails
- Cache all received events in localStorage for persistence
- On reconnect: load any missed events from server

### Acceptance Criteria:
- Reconnection works after 30-second network outage
- User is informed when connection lost
- Cached data is always available (even if offline)
- No lost dive data
- Graceful degradation (no errors in console)

### Files to Modify:
- `diveanalyzer/server/sse_server.py` - Add event caching/history endpoint
- `diveanalyzer/utils/review_gallery.py` - Add reconnection logic and caching

---

## Event Payload Examples

### dive_detected
```json
{
  "event": "dive_detected",
  "dive_index": 5,
  "dive_id": "dive_005",
  "start_frame": 450,
  "splash_frame": 520,
  "end_frame": 600,
  "duration": 1.23,
  "confidence": 0.94,
  "phase": "phase_1"
}
```

### thumbnail_ready
```json
{
  "event": "thumbnail_ready",
  "dive_id": "dive_005",
  "type": "grid",
  "frames": ["data:image/jpeg;base64,...", "..."],
  "generated_at": "2026-01-21T10:30:45Z"
}
```

### status_update
```json
{
  "event": "status_update",
  "phase": "audio_detection",
  "progress": 0.65,
  "message": "🔍 Detecting audio peaks... (42/60 dives found)",
  "dives_found": 42,
  "dives_expected": 60,
  "processing_speed": 2.1
}
```

### processing_complete
```json
{
  "event": "processing_complete",
  "total_dives": 61,
  "total_thumbnails_generated": 549,
  "total_time_seconds": 145,
  "completed_at": "2026-01-21T10:32:30Z"
}
```

---

## Implementation Order

1. **FEAT-01** - HTTP server foundation (all others depend on this)
2. **FEAT-02** - HTML event consumer (needed for any updates)
3. **FEAT-06** - Auto-launch browser (cosmetic but good UX for testing)
4. **FEAT-03** - Placeholder system (shows immediate progress)
5. **FEAT-05** - Status dashboard (visibility into processing)
6. **FEAT-07** - Deferred thumbnails (main UX improvement)
7. **FEAT-04** - Progressive thumbnail loading (integrates with feat-07)
8. **FEAT-08** - Connection management (robustness layer)

---

## Testing Strategy

For each ticket:
- Unit tests for event generation
- Integration test with mock HTML client
- Manual test with real video (IMG_6497.MOV)
- Network disruption test (kill server during processing)
- Performance test (ensure no jank during concurrent updates)

---

## Success Metrics

- **Time to gallery appearance**: < 3 seconds from CLI start
- **Time to first interaction**: User can select/delete dives while thumbnails generate
- **Thumbnail arrival**: First thumbnail within 30 seconds, all within 2 minutes
- **Smoothness**: No UI jank or freezing during updates
- **Robustness**: Zero lost dive data on network interruption

