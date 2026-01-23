# FEAT-02: Technical Implementation Guide

## Code Structure

### EventStreamConsumer Class

**Location**: `diveanalyzer/utils/review_gallery.py` (lines 1176-1392 in script section)

```javascript
class EventStreamConsumer {
    // Constructor
    constructor(serverUrl = null)

    // Public methods
    connect()                           // Establish connection
    disconnect()                        // Close connection
    getLogEntries()                    // Get all logged events
    clearLog()                         // Clear event log

    // Private methods
    _detectServerUrl()                 // Auto-detect server URL
    _handleEvent(eventType, event)     // Process received event
    _handleConnectionError()           // Handle connection failures
    _updateLatestEvent(eventType, data) // Update status bar
    _updateStatus(status, message)     // Update connection indicator
    _logEvent(type, message, logType)  // Add event to log
    _renderEventLog()                  // Render log to DOM
    _getEventLogType(eventType)        // Determine event category
}
```

### Public API

#### `constructor(serverUrl = null)`

Initialize the event consumer.

**Parameters:**
- `serverUrl` (string, optional): Server URL. If not provided, auto-detects from window.location

**Example:**
```javascript
// Auto-detect
const consumer = new EventStreamConsumer()

// Explicit URL
const consumer = new EventStreamConsumer('http://my-server.com:8765')
```

#### `connect()`

Establish SSE connection to server.

**Behavior:**
- Creates EventSource connection to `{serverUrl}/events`
- Registers listeners for all event types
- Updates status to "connected" on success
- Triggers reconnection on error

**Example:**
```javascript
consumer.connect()
```

#### `disconnect()`

Close the SSE connection.

**Behavior:**
- Closes EventSource
- Updates status to "disconnected"
- Logs disconnect event

**Example:**
```javascript
consumer.disconnect()
```

#### `getLogEntries()`

Retrieve all logged events.

**Returns:** Array of event objects

**Example:**
```javascript
const events = consumer.getLogEntries()
events.forEach(e => console.log(e.timestamp, e.message))
```

#### `clearLog()`

Clear the event log and re-render UI.

**Example:**
```javascript
consumer.clearLog()
```

### Private Methods

#### `_detectServerUrl()`

Auto-detect server URL based on execution context.

**Logic:**
- If `window.location.hostname === 'file:'` (local file) → `http://localhost:8765`
- Otherwise → `http://{window.location.hostname}:8765`

#### `_handleEvent(eventType, event)`

Process a received SSE event.

**Steps:**
1. Parse JSON data from event.data
2. Call `_updateLatestEvent()` to update status bar
3. Call `_logEvent()` to add to event log
4. Log to browser console

#### `_handleConnectionError()`

Handle connection failures and attempt reconnection.

**Steps:**
1. Set `isConnected = false`
2. Update status to "disconnected"
3. If attempts < 5:
   - Increment `reconnectAttempts`
   - Calculate delay (exponential backoff)
   - Schedule reconnection via `setTimeout()`
4. Else:
   - Update status to "Server not available"
   - Log warning

**Backoff Formula:**
```
delay = reconnectDelay * reconnectAttempts
     = 2000ms * 1 = 2s
     = 2000ms * 2 = 4s
     = 2000ms * 3 = 6s
     = 2000ms * 4 = 8s
     = 2000ms * 5 = 10s
```

#### `_updateLatestEvent(eventType, data)`

Update the status text with latest event.

**Uses:** `requestAnimationFrame()` for non-blocking update

**Updates:**
- Status text shows: `[HH:MM:SS] event_type`

#### `_updateStatus(status, message)`

Update connection status indicator.

**Parameters:**
- `status`: 'connected', 'disconnected', or 'connecting'
- `message`: Display message

**Updates:**
- Status div class
- Indicator color and animation
- Status text
- Server address

**Uses:** `requestAnimationFrame()` for smooth update

#### `_logEvent(type, message, logType)`

Add event to the event log.

**Parameters:**
- `type`: Event type name
- `message`: Display message
- `logType`: 'info', 'success', 'warning', or 'error'

**Behavior:**
- Creates event object with timestamp
- Adds to `eventLog` array
- If > 100 entries, removes oldest
- Calls `_renderEventLog()`

#### `_renderEventLog()`

Render event log to DOM.

**Uses:** `requestAnimationFrame()` for batch updates

**Generates HTML:**
```html
<div class="event-log-entry {logType}">
    <span class="event-log-timestamp">{timestamp}</span>
    <span>{message}</span>
</div>
```

**Behavior:**
- Replaces `#eventLog` innerHTML
- Auto-scrolls to bottom

#### `_getEventLogType(eventType)`

Determine the CSS class for an event type.

**Mapping:**
- 'connected' → 'success' (green)
- 'error' → 'error' (red)
- '*_complete', '*_detected' → 'info' (blue)
- Other → 'info' (blue)

### Initialization

#### `initializeEventConsumer()`

Initialize the event consumer and UI controls.

**Steps:**
1. Detect server URL
2. Create EventStreamConsumer instance
3. Set up toggle button listener
4. Set up close button listener
5. Call `connect()`

**Location:** Line 1397-1438 in script section

**Invoked from:** `initGallery()` at line 1447

### HTML Elements

#### Status Indicator (`#connectionStatus`)

```html
<div class="connection-status" id="connectionStatus">
    <span class="status-indicator connecting"></span>
    <span id="statusText">Connecting...</span>
    <span class="status-address" id="statusAddress"></span>
</div>
```

**CSS Classes:**
- `connected` - Green, connected state
- `disconnected` - Red, disconnected state
- `connecting` - Yellow, connecting/reconnecting state

**Child Elements:**
- `.status-indicator` - Animated colored dot
- `#statusText` - Status message
- `#statusAddress` - Server address

#### Event Log Container (`#eventLogContainer`)

```html
<div class="event-log-container" id="eventLogContainer">
    <div class="event-log-header">
        <span>Live Events</span>
        <button class="event-log-close" id="closeEventLog">&times;</button>
    </div>
    <div class="event-log" id="eventLog"></div>
</div>
```

**CSS Classes:**
- `show` - Display container
- (no class) - Hidden

**Child Elements:**
- `.event-log-header` - Title and close button
- `#closeEventLog` - Close button
- `#eventLog` - Event list container

#### Toggle Button (`#toggleEventLog`)

```html
<button class="toggle-event-log" id="toggleEventLog" title="Show/hide live events">📋</button>
```

**Behavior:**
- Click to toggle `.show` class on `#eventLogContainer`
- Hidden when log is open
- Visible when log is closed

### CSS Styles

#### Status Indicator

```css
.connection-status {
    position: fixed;
    top: 20px;
    right: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 15px;
    border-radius: 8px;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    z-index: 500;
}

.status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
```

**Colors:**
- `.connected .status-indicator` - `#4CAF50` (green)
- `.disconnected .status-indicator` - `#f44336` (red)
- `.connecting .status-indicator` - `#FF9800` (orange)

#### Event Log

```css
.event-log-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 350px;
    max-height: 300px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 499;
    display: none;
    flex-direction: column;
}

.event-log-container.show {
    display: flex;
}

.event-log-entry {
    padding: 6px 8px;
    margin-bottom: 6px;
    border-radius: 3px;
    border-left: 3px solid;
}

.event-log-entry.info { border-left-color: #2196F3; }
.event-log-entry.success { border-left-color: #4CAF50; }
.event-log-entry.warning { border-left-color: #FF9800; }
.event-log-entry.error { border-left-color: #f44336; }
```

## Event Flow Diagrams

### Normal Operation

```
Browser loads page
    ↓
DOMContentLoaded
    ↓
initGallery()
    ↓
initializeEventConsumer()
    ↓
EventStreamConsumer.connect()
    ↓
EventSource('/events') opens
    ↓
Server sends 'connected' event
    ↓
_handleEvent('connected')
    ↓
Three parallel:
├─ _updateLatestEvent() ──→ DOM update via rAF
├─ _logEvent() ───────────→ Add to array
└─ _renderEventLog() ─────→ Render via rAF
    ↓
Status indicator: Green ✓
Event log: Shows "Connected to server"
```

### Connection Error

```
EventSource error event
    ↓
onerror handler triggered
    ↓
_handleConnectionError()
    ↓
reconnectAttempts < 5?
    ├─ YES: Schedule reconnection with backoff
    │       Update status: "Reconnecting... (1/5)"
    │       Wait 2 seconds
    │       Retry connect()
    │
    └─ NO:  Max retries exceeded
            Status: "Server not available"
            Log: "using local mode"
            Gallery still functional
```

### Event Processing

```
Server emits: splash_detection_complete
    ↓
SSE sends: event: splash_detection_complete
          data: {"peak_count": 5, "timestamp": "..."}
    ↓
EventSource fires event
    ↓
addEventListener('splash_detection_complete') called
    ↓
_handleEvent('splash_detection_complete', event)
    ↓
JSON.parse(event.data) → {peak_count: 5, ...}
    ↓
_updateLatestEvent() → Status: "12:34:56 splash_detection_complete"
_logEvent() → Log entry added
    ↓
_renderEventLog() → DOM updated
    ↓
Browser console: "SSE: Event received - splash_detection_complete"
```

## Integration Points

### With Gallery

**Location:** `initGallery()` function

```javascript
function initGallery() {
    // ... existing code ...

    // FEAT-02: Initialize real-time event consumer
    initializeEventConsumer();

    // ... rest of initialization ...
}
```

### With Server (FEAT-01)

The server must:
1. Be running on `localhost:8765` (or configured URL)
2. Provide `/events` SSE endpoint
3. Send events in format:
   ```
   event: {event_type}
   data: {JSON payload}
   ```

Example server emit call:
```python
server.emit('dive_detected', {'dive_id': 1, 'confidence': 0.95})
```

Results in SSE:
```
event: dive_detected
data: {"event_type": "dive_detected", "timestamp": "2024-01-21T...", "payload": {"dive_id": 1, "confidence": 0.95}}
```

## Debugging Tips

### Browser Console Commands

```javascript
// Check connection status
console.log('Connected:', eventConsumer.isConnected)
console.log('Server:', eventConsumer.serverUrl)
console.log('Reconnect attempts:', eventConsumer.reconnectAttempts)

// View all events
console.log(eventConsumer.getLogEntries())
console.table(eventConsumer.getLogEntries())

// Manually emit event (simulate server)
eventConsumer._logEvent('test', 'Manual test event', 'info')

// Force disconnect
eventConsumer.disconnect()

// Manual reconnect
eventConsumer.reconnectAttempts = 0
eventConsumer.connect()

// Export events as JSON
const json = JSON.stringify(eventConsumer.getLogEntries(), null, 2)
console.log(json)
```

### Expected Console Output

**Successful connection:**
```
SSE: Attempting to connect to http://localhost:8765/events
SSE: Connection opened
SSE: Event received - connected: {message: "SSE stream established"}
```

**Connection error:**
```
SSE: Attempting to connect to http://localhost:8765/events
SSE: Connection error: NetworkError: ...
SSE: Attempting to reconnect in 2000ms (attempt 1/5)
SSE: Attempting to reconnect in 4000ms (attempt 2/5)
... (up to 5 attempts)
SSE: Max reconnection attempts exceeded
```

**Server unavailable:**
```
Connection status: "Server not available"
Event log: "using local mode"
Gallery continues to function normally
```

## Performance Considerations

### Memory Usage

- Event log limited to 100 entries
- Each entry: ~500 bytes
- Max memory: ~50KB for event log
- EventSource connection: Minimal (~1KB state)

### CPU Usage

- Idle: ~0% (event listener waiting)
- On event: ~0.1-0.5% (parsing + DOM update)
- Reconnection: ~0% (async setTimeout)

### Network Usage

- Connection: ~1KB (HTTP headers)
- Per event: ~200 bytes (typical)
- Keepalive: 30 byte comment every 30s
- Total for 100 events: ~20KB

## Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome  | ✓ Full  | EventSource since 2.0 |
| Firefox | ✓ Full  | EventSource since 6.0 |
| Safari  | ✓ Full  | EventSource since 5.0 |
| Edge    | ✓ Full  | EventSource since 15.0 |
| IE 11   | ✗ No    | No EventSource support |

Graceful degradation: Gallery works without events on unsupported browsers.

## Known Limitations

1. **CORS**: May require CORS headers if server on different origin
2. **EventSource**: Cannot set custom headers (limitation of API)
3. **Reconnection**: Manual only if max attempts reached
4. **History**: Events not persisted between sessions
5. **Filtering**: All events received (no server-side filtering)

## Future Enhancements

1. **Custom Server URL**: Configuration UI to change server address
2. **Event Filtering**: Only subscribe to specific event types
3. **Persistent Logging**: Save events to localStorage
4. **Export**: Download event log as CSV/JSON
5. **Replay**: Replay event log with timing
6. **Notifications**: Browser notifications for key events
7. **Metrics**: Event count, frequency statistics
8. **Search**: Search/filter events by type or content

## Testing Checklist

- [ ] Connection indicator updates on load
- [ ] Status shows "Connected" when server available
- [ ] Status shows "Disconnected" when server unavailable
- [ ] Event log displays events with timestamps
- [ ] Colors match event types
- [ ] Auto-scrolls to latest event
- [ ] Toggle button shows/hides log
- [ ] Gallery functions without server
- [ ] Reconnection works (disconnect server, restart)
- [ ] Multiple events in rapid succession handled
- [ ] Memory doesn't grow unbounded
- [ ] No console errors
- [ ] Responsive on mobile
- [ ] Keyboard shortcuts still work
- [ ] Modal still functions with events

## References

- MDN: EventSource API - https://developer.mozilla.org/en-US/docs/Web/API/EventSource
- MDN: Server-Sent Events - https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- W3C: SSE Spec - https://html.spec.whatwg.org/multipage/server-sent-events.html

## Related Features

- **FEAT-01**: HTTP Server with SSE endpoint
- **FEAT-03**: Server-side event filtering
- **FEAT-04**: Historical event replay
- **FEAT-05**: Event persistence
