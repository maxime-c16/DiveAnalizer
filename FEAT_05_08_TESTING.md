# FEAT-05 & FEAT-08 Testing Guide

## Overview

This document provides comprehensive testing instructions for:
- **FEAT-05: Status Dashboard & Progress Tracking** - Real-time processing status display with metrics
- **FEAT-08: Connection Management & Fallback** - SSE reconnection with exponential backoff and polling fallback

## Features Implemented

### FEAT-05: Status Dashboard

The status dashboard provides real-time monitoring with:

#### Visual Elements
- **Sticky header** at top of review gallery (z-index 600)
- **Phase indicator** with color coding:
  - Phase 1 (Audio): Blue (#3b82f6)
  - Phase 2 (Motion): Yellow (#eab308)
  - Phase 3 (Person): Green (#22c55e)
- **Progress bar** with smooth animation (0-100%)
- **Four key metrics**:
  1. Dives Found (X/Y format showing detected vs expected)
  2. Processing Speed (dives/minute)
  3. Time Remaining (HH:MM format, auto-calculated)
  4. Thumbnails Ready (X/Y format)

#### Event Payload Format
```json
{
  "event": "status_update",
  "phase": "phase_1",
  "phase_name": "Audio Detection",
  "dives_found": 42,
  "dives_expected": 60,
  "thumbnails_ready": 15,
  "thumbnails_expected": 488,
  "elapsed_seconds": 45,
  "processing_speed": 0.93,
  "progress_percent": 65
}
```

### FEAT-08: Connection Management

Advanced connection handling with:

#### Reconnection Strategy
- **Exponential backoff**: 1s → 2s → 4s → 8s → 8s
- **Max 5 reconnection attempts**
- **Visual status**: Shows "Reconnecting... (X/5)" during attempts
- **Fallback to polling**: After all reconnect attempts fail
- **Manual retry button**: Available in connection lost state

#### Event Caching
- **localStorage persistence**: Stores up to 500 events
- **Automatic deduplication**: Prevents duplicate events when polling
- **Survives page reloads**: Events loaded on page refresh
- **Graceful degradation**: Gallery works offline with cached data

#### Connection Banner
- **Sticky banner** below status dashboard (z-index 599)
- **Two states**:
  - Reconnecting (orange, with spinner)
  - Connection Lost (red, with retry button)
- **Auto-dismisses** on successful reconnection

---

## Testing Procedures

### Test 1: Status Dashboard Display

**Objective**: Verify dashboard renders and displays correctly

**Steps**:
1. Start a dive analysis: `diveanalyzer process video.mov --enable-server`
2. Open browser to `http://localhost:8765`
3. Wait for Phase 1 (audio detection) to complete

**Expected Results**:
- [ ] Dashboard visible at top with purple gradient background
- [ ] Phase indicator shows "Phase 1 - Audio Detection" in blue
- [ ] Four metrics visible with correct formatting
- [ ] Progress bar shows 0% initially, then animates to 33%
- [ ] Dashboard is sticky (stays visible when scrolling)

**Terminal Output**:
```
✓ Server running at http://localhost:8765
✓ Events: http://localhost:8765/events
```

---

### Test 2: Phase Transitions

**Objective**: Verify phase changes and color updates

**Steps**:
1. Run analysis with motion detection: `diveanalyzer process video.mov --enable-server --enable-motion`
2. Watch phase indicator through all phases

**Expected Results**:
- [ ] Phase transitions: Phase 1 (blue) → Phase 2 (yellow) → Phase 3 (green/if enabled)
- [ ] Phase label updates: "Audio Detection" → "Motion Detection" → "Person Detection"
- [ ] Progress bar increments: 33% → 66% → 90% → 100%
- [ ] Each transition triggers smooth color change (no jank)

---

### Test 3: Metrics Updates

**Objective**: Verify all metrics display accurate information

**Steps**:
1. Run: `diveanalyzer process video.mov --enable-server --enable-motion --enable-person`
2. Monitor dashboard throughout processing

**Expected Results**:
- [ ] **Dives Found**: Updates after each detection (e.g., 5/50, 10/50, 20/50)
- [ ] **Processing Speed**: Displays reasonable values (0.5-2.0 dives/min)
- [ ] **Time Remaining**: Updates as dives_found increases
  - Formula: `(dives_expected - dives_found) / processing_speed` (in minutes)
  - Display format: "M:SS" (e.g., "5:30", "0:45")
- [ ] **Thumbnails Ready**: Shows progress of thumbnail generation

**Calculation Check** (Manual):
```
If: dives_found=30, dives_expected=60, processing_speed=2.0 dives/min
Then: remaining=(60-30)/2.0=15 minutes → displays "15:00"
```

---

### Test 4: Exponential Backoff Reconnection

**Objective**: Verify reconnection behavior with exponential delays

**Steps**:
1. Start server: `diveanalyzer process video.mov --enable-server`
2. Open browser to dashboard
3. Kill server: `Ctrl+C` in terminal
4. Monitor connection banner and browser console

**Expected Results - Connection Banner**:
- [ ] Banner appears with "Reconnecting... (1/5)"
- [ ] Spinner visible (animated loading circle)
- [ ] No retry button initially

**Expected Results - Browser Console** (F12 → Console):
```
SSE: Attempting to connect to http://localhost:8765/events (attempt 1/5)
[Wait ~1 second]
SSE: Connection error
SSE: Attempting to reconnect in 1000ms (attempt 1/5)
[Wait ~2 seconds]
SSE: Connection error
SSE: Attempting to reconnect in 2000ms (attempt 2/5)
[Wait ~4 seconds]
SSE: Connection error
SSE: Attempting to reconnect in 4000ms (attempt 3/5)
[Wait ~8 seconds]
SSE: Connection error
SSE: Attempting to reconnect in 8000ms (attempt 4/5)
[Wait ~8 seconds]
SSE: Connection error
SSE: Attempting to reconnect in 8000ms (attempt 5/5)
[Wait ~8 seconds]
SSE: Connection error
SSE: Max reconnection attempts exceeded
SSE: Starting fallback polling every 3 seconds
```

**Reconnection Banner States**:
- Attempt 1: "Reconnecting... (1/5)"
- Attempt 2: "Reconnecting... (2/5)"
- Attempt 3: "Reconnecting... (3/5)"
- Attempt 4: "Reconnecting... (4/5)"
- Attempt 5: "Reconnecting... (5/5)"
- After 5: "Connection lost - Using cached data" + red background + visible retry button

**Timing Verification**:
- 1st attempt fails ~1s after initial connection attempt
- 2nd attempt fails ~3s after that (1s delay + 2s attempt)
- 3rd attempt fails ~6s after that (2s delay + 4s attempt)
- 4th attempt fails ~12s after that (4s delay + 8s attempt)
- 5th attempt fails ~16s after that (8s delay + 8s attempt)

---

### Test 5: Polling Fallback

**Objective**: Verify fallback to polling after SSE failure

**Steps**:
1. Same setup as Test 4
2. After all 5 reconnection attempts fail, observe polling
3. Start server again in a new terminal
4. Verify reconnection from polling

**Expected Results - Polling Phase**:
- [ ] Connection banner shows error state (red, with retry button)
- [ ] Browser console shows: `SSE: Starting fallback polling every 3 seconds`
- [ ] Network tab (F12 → Network) shows requests to `/events-history` every 3 seconds
- [ ] Each polling request returns `{events: [], count: 0}`
- [ ] No errors, polling continues smoothly

**Expected Results - Reconnection from Polling**:
1. Start server: `diveanalyzer process video.mov --enable-server` (in new terminal)
2. Observe in browser:
   - [ ] Within ~3 seconds: `/events-history` returns new events
   - [ ] New events are processed (status updates rendered)
   - [ ] After server is fully ready, /events endpoint becomes available
   - [ ] SSE connection switches from polling back to live streaming
   - [ ] Console shows: `SSE: Connected to server`
   - [ ] Connection banner disappears
   - [ ] Dashboard updates in real-time (not every 3s)

---

### Test 6: localStorage Persistence

**Objective**: Verify events are cached and survive page reload

**Steps**:
1. Start server: `diveanalyzer process video.mov --enable-server`
2. Open browser, wait for several status updates
3. Open DevTools: F12 → Application → Local Storage → `localhost:8765`
4. Verify key `diveanalyzer_events` exists
5. Reload page: F5

**Expected Results**:
- [ ] **Before reload**: `diveanalyzer_events` key visible in localStorage
- [ ] **localStorage content**: Valid JSON array with event objects
  ```json
  [
    {
      "event_type": "status_update",
      "data": {...},
      "timestamp": "2026-01-21T15:30:45.123Z"
    },
    ...
  ]
  ```
- [ ] **Max 500 events**: If > 500 events, oldest ones removed
- [ ] **After reload**:
  - Dashboard loads immediately (doesn't say "Waiting for dives...")
  - Cached events processed (tiles might show cached dives)
  - New live events continue streaming
  - No data loss

---

### Test 7: Connection Lost with Cached Data

**Objective**: Verify gallery works offline with cached data

**Steps**:
1. Start server, process some dives
2. Kill server while processing
3. Try interacting with cached dives on page

**Expected Results**:
- [ ] Connection banner shows error state (red)
- [ ] All cached events still visible
- [ ] Can toggle dives for deletion/keeping
- [ ] Can watch selected dives (playback still works)
- [ ] Can select/deselect dives
- [ ] All interactive features work without server
- [ ] Cannot send actions to server (expected, shown as disabled state)

---

### Test 8: Manual Retry Button

**Objective**: Verify manual reconnection via retry button

**Steps**:
1. Kill server (connection lost state)
2. Retry button visible in red banner
3. Click retry button
4. Start server in another terminal

**Expected Results**:
- [ ] Retry button responsive (not disabled)
- [ ] After click: `reconnectAttempts` reset to 0 (console: `Manual reconnect requested`)
- [ ] Connection attempts start fresh from attempt 1/5
- [ ] Banner shows "Reconnecting... (1/5)" instead of error
- [ ] If server is already running: connection succeeds immediately
- [ ] Banner disappears, dashboard resumes live updates

---

### Test 9: Retry Button Disabled States

**Objective**: Verify retry button availability

**Steps**:
1. Connected state: No error banner, no retry button ✓
2. Reconnecting (1-4): Error banner visible, retry button hidden
3. Connection lost (after 5 attempts): Error banner visible, retry button visible

**Expected Results**:
- [ ] Retry button hidden during reconnection attempts
- [ ] Retry button only visible in final error state
- [ ] Button clickable and responsive

---

### Test 10: Browser Console Logging

**Objective**: Verify detailed debugging output

**Steps**:
1. Open DevTools: F12 → Console
2. Set log level: `console.log('SSE: Debug mode enabled')`
3. Run full processing cycle with connection loss

**Expected Console Output**:
```javascript
// Connection phase
SSE: Attempting to connect to http://localhost:8765/events
SSE: Connection opened
SSE: Connected to server
[Events stream in]
SSE: Event received - status_update: {phase: "phase_1", ...}

// Connection loss
SSE: Connection error
SSE: Attempting to reconnect in 1000ms (attempt 1/5)
[etc...]

// Polling phase
SSE: Starting fallback polling every 3 seconds
[Polling requests...]

// Recovery
SSE: Loaded 25 cached events from localStorage
[New events come in via polling]
```

---

## Manual Testing via Browser Console

### Simulate Status Updates

Test the dashboard without processing:

```javascript
// Open DevTools Console and paste:
if (statusDashboard) {
  statusDashboard.update({
    phase: "phase_1",
    phase_name: "Audio Detection",
    dives_found: 15,
    dives_expected: 50,
    thumbnails_ready: 5,
    thumbnails_expected: 150,
    processing_speed: 0.5,
    elapsed_seconds: 30,
    progress_percent: 30
  });
}
```

Expected: Dashboard metrics update in real-time

### Simulate Connection States

```javascript
// Simulate connection lost
if (eventConsumer) {
  eventConsumer._showConnectionBanner('reconnecting', 3);
}

// Simulate connection error (final state)
if (eventConsumer) {
  eventConsumer._showConnectionBanner('error', 5);
}

// Hide banner
if (eventConsumer) {
  eventConsumer._hideConnectionBanner();
}
```

### Check Cached Events

```javascript
// View cached events
console.log(JSON.stringify(eventConsumer.cachedEvents, null, 2));

// Clear cache
localStorage.removeItem('diveanalyzer_events');
console.log('Cache cleared');

// Count events
console.log('Cached events: ' + eventConsumer.cachedEvents.length);
```

---

## Performance Benchmarks

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Dashboard Update Latency | <16ms | 60fps animations |
| Status Update Frequency | 500-1000ms | CLI emits every ~1s |
| Memory (cached events) | <5MB | 500 events max |
| localStorage Write Time | <10ms | JSON stringification |
| CPU Usage (idle dashboard) | <1% | Minimal background work |
| CSS Animation FPS | 60 FPS | Smooth progress bar |

### Testing CPU/Memory

```javascript
// In DevTools Console:
// Monitor memory
console.memory.usedJSHeapSize

// Monitor CPU (Chrome DevTools Performance tab)
// Record 10 seconds while dashboard updating
```

---

## Accessibility Testing

- [ ] Tab navigation through retry button
- [ ] Keyboard focus visible on retry button
- [ ] Color contrast meets WCAG AA (dashboard text vs background)
- [ ] Progress bar accessible via screen reader
- [ ] Status text descriptive (not just colored indicators)

---

## Browser Compatibility

| Browser | Version | Status | Notes |
|---------|---------|--------|-------|
| Chrome | Latest | ✓ | Full support |
| Safari | Latest | ✓ | Full support |
| Firefox | Latest | ✓ | Full support |
| Edge | Latest | ✓ | Full support |
| Mobile Safari | Latest | ✓ | Responsive design tested |

---

## Edge Cases

### Edge Case 1: Multiple Page Reloads

```javascript
// Reload 5 times in rapid succession
for(let i=0; i<5; i++) {
  setTimeout(() => location.reload(), i*1000);
}
// Expected: Cache loads properly each time, no data corruption
```

### Edge Case 2: Large Event Backlog

```javascript
// Simulate server emitting 100 events rapidly
for(let i=0; i<100; i++) {
  if(server) {
    server.emit("status_update", {
      progress_percent: i,
      dives_found: i,
      // ... other fields
    });
  }
}
// Expected: Dashboard handles without stalling, cache limited to 500
```

### Edge Case 3: Simultaneous Polling and SSE

During the transition from polling back to SSE:
- [ ] No duplicate events processed
- [ ] Dashboard updates smoothly during transition
- [ ] Polling requests stop after SSE reconnects
- [ ] Memory stays bounded

### Edge Case 4: localStorage Full

```javascript
// Fill localStorage beyond capacity
const largeData = 'x'.repeat(5*1024*1024);
localStorage.setItem('test', largeData);
// Now try to cache an event
// Expected: Graceful error handling, console warning
```

---

## Deployment Checklist

- [ ] StatusDashboard class minified and included
- [ ] EventStreamConsumer reconnection logic tested
- [ ] localStorage caching verified
- [ ] /events-history endpoint responding correctly
- [ ] Status updates emitted at correct phases
- [ ] Browser console logging clear and helpful
- [ ] CSS animations smooth (no jank)
- [ ] Mobile responsive (tested on 320px - 1920px widths)
- [ ] Accessibility passed WCAG checks
- [ ] All console.log statements appropriate
- [ ] No memory leaks (profiled with DevTools)

---

## Troubleshooting

### Dashboard Not Updating

```javascript
// Check if statusDashboard exists
console.log(statusDashboard);

// Check if events are being received
console.log(eventConsumer.eventLog);

// Check if cache has events
console.log(eventConsumer.cachedEvents);
```

### Connection Banner Stuck

```javascript
// Reset connection state
eventConsumer.reconnectAttempts = 0;
eventConsumer._hideConnectionBanner();
eventConsumer.connect();
```

### Polling Not Working

```javascript
// Check if polling is active
console.log(eventConsumer.pollingActive);

// Check /events-history endpoint
fetch('http://localhost:8765/events-history')
  .then(r => r.json())
  .then(d => console.log(d));
```

### Cache Corrupted

```javascript
// Clear and reinitialize
localStorage.removeItem('diveanalyzer_events');
location.reload();
```

---

## Final Verification

Run this checklist before deployment:

1. [ ] Status dashboard renders in all phases
2. [ ] Progress bar animates smoothly 0-100%
3. [ ] All 4 metrics display correctly
4. [ ] Phase colors correct (blue/yellow/green)
5. [ ] Exponential backoff timing verified (1s, 2s, 4s, 8s, 8s)
6. [ ] Reconnection banner shows attempt count (X/5)
7. [ ] Polling fallback works after 5 attempts
8. [ ] Retry button available and functional
9. [ ] localStorage persists through page reload
10. [ ] Cached events deduped during polling
11. [ ] No console errors in any phase
12. [ ] CPU/Memory usage stays below targets
13. [ ] Mobile responsive design verified
14. [ ] Accessibility WCAG AA compliant
15. [ ] All browser compatibility tested

---

## References

- **Status Update Event**: Emitted every ~1 second during processing
- **Reconnection Delays**: [1s, 2s, 4s, 8s, 8s]
- **localStorage Key**: `diveanalyzer_events`
- **Max Cached Events**: 500
- **Polling Interval**: 3 seconds
- **Progress Stages**: 33% (Phase 1) → 66% (Phase 2) → 90% (Phase 3) → 100% (Complete)

