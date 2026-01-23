# FEAT-01: HTTP Server & SSE Endpoint Integration

## Overview

This feature implements a lightweight HTTP server with Server-Sent Events (SSE) support for real-time dive review. The server runs in a background thread during video processing, providing live updates to connected clients without blocking the detection pipeline.

## Architecture

### Core Components

1. **EventServer** (`diveanalyzer/server/sse_server.py`)
   - Manages HTTP server lifecycle (start/stop)
   - Runs in background thread (non-blocking)
   - Emits real-time events to connected clients
   - Graceful shutdown with proper cleanup

2. **EventQueue**
   - Thread-safe event queue with multiple subscriber support
   - Maintains list of active SSE connections
   - Publishes events to all subscribers atomically

3. **DiveReviewSSEHandler**
   - Custom HTTP request handler
   - Implements SSE protocol for `/events` endpoint
   - Serves HTML gallery at root endpoint
   - Provides health check endpoint

### HTTP Endpoints

```
GET /
  Serves the review gallery HTML file
  Returns: HTML content (text/html)

GET /events
  Server-Sent Events stream for live updates
  Returns: Continuous stream of JSON events (text/event-stream)

GET /health
  Health check endpoint
  Returns: {"status": "ok", "timestamp": "..."}
```

## Usage

### Basic Server Usage

```python
from diveanalyzer.server import EventServer

# Create server instance
server = EventServer(
    gallery_path="/path/to/review_gallery.html",
    host="localhost",
    port=8765,
    log_level="INFO"
)

# Start server in background
if server.start():
    print(f"Server running at {server.get_url()}")

    # Emit events during processing
    server.emit("dive_detected", {
        "dive_id": 1,
        "confidence": 0.95,
        "duration": 1.2
    })

    # Graceful shutdown
    server.stop()
```

### CLI Integration

Enable the server with `--enable-server` flag:

```bash
# Start server on default port (8765)
diveanalyzer process video.mp4 --enable-server

# Use custom port
diveanalyzer process video.mp4 --enable-server --server-port 9000
```

The server will:
1. Start before video processing
2. Emit events during detection phases
3. Serve the review gallery at `http://localhost:8765`
4. Shutdown gracefully after processing

## Event Types

The server emits the following event types during processing:

### Connection Events
```json
{
  "event_type": "connected",
  "timestamp": "2026-01-21T12:34:56.789Z",
  "payload": {
    "message": "SSE stream established"
  }
}
```

### Detection Events

**Splash Detection Complete**
```json
{
  "event_type": "splash_detection_complete",
  "payload": {
    "peak_count": 5,
    "threshold_db": -25.0
  }
}
```

**Motion Detection Complete**
```json
{
  "event_type": "motion_detection_complete",
  "payload": {
    "burst_count": 4,
    "proxy_height": 480
  }
}
```

**Person Detection Complete**
```json
{
  "event_type": "person_detection_complete",
  "payload": {
    "departure_count": 3,
    "confidence_threshold": 0.5
  }
}
```

### Dive Events

**Dives Detected**
```json
{
  "event_type": "dives_detected",
  "payload": {
    "dive_count": 5,
    "signal_type": "audio + motion",
    "confidence_threshold": 0.5
  }
}
```

**Extraction Complete**
```json
{
  "event_type": "extraction_complete",
  "payload": {
    "total_dives": 5,
    "successful": 5,
    "failed": 0
  }
}
```

### Completion Events

**Processing Complete**
```json
{
  "event_type": "processing_complete",
  "payload": {
    "status": "success",
    "output_directory": "/path/to/dives"
  }
}
```

## Client Integration

### JavaScript Client

```javascript
// Connect to SSE stream
const eventSource = new EventSource('http://localhost:8765/events');

// Handle connection
eventSource.addEventListener('connected', (e) => {
  console.log('Connected to server:', JSON.parse(e.data));
});

// Handle dive detection
eventSource.addEventListener('dives_detected', (e) => {
  const data = JSON.parse(e.data);
  console.log(`Detected ${data.dive_count} dives`);
});

// Handle extraction complete
eventSource.addEventListener('extraction_complete', (e) => {
  const data = JSON.parse(e.data);
  console.log(`Extracted ${data.successful}/${data.total_dives} clips`);
});

// Handle errors
eventSource.onerror = () => {
  console.error('Connection to server failed');
};
```

### Python Client

```python
import json
import sseclient

# Connect to SSE stream
url = 'http://localhost:8765/events'
response = requests.get(url, stream=True)
client = sseclient.SSEClient(response)

# Process events
for event in client:
    if event.event:
        data = json.loads(event.data)
        print(f"Event: {event.event}")
        print(f"Data: {data}")
```

## Technical Details

### Thread Safety

- **EventQueue**: Uses `threading.Lock()` for subscriber list management
- **Subscriber Queues**: Uses `queue.Queue` for thread-safe event delivery
- **Handler State**: Class variables set before server start

### Non-Blocking Architecture

1. **Background Thread**: Server runs in daemon thread, doesn't block main process
2. **Timeout Handling**: `handle_request()` uses 1.0s socket timeout for graceful shutdown
3. **Async Event Publishing**: Events published without waiting for subscriber acknowledgment

### Graceful Shutdown

1. `server.stop()` sets `running = False`
2. Server thread detects flag and exits main loop
3. Socket cleanup occurs in `finally` block
4. Thread joins with 5s timeout before returning

### Error Handling

- Broken pipe errors logged as debug (expected on client disconnect)
- Failed event publishes to dead subscribers logged and subscriber removed
- Server startup failures caught and reported to CLI
- Cleanup occurs even if errors encountered

## Testing

Run the test suite:

```bash
python test_sse_server.py
```

Tests include:
- Server startup and shutdown
- HTTP endpoints (health, gallery)
- SSE event stream connection
- Event emission and reception
- Multiple concurrent subscribers
- URL generation methods

## Configuration

### Server Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gallery_path` | str | Required | Path to review gallery HTML file |
| `host` | str | "localhost" | Host to bind to |
| `port` | int | 8765 | Port to listen on |
| `log_level` | str | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |

### CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable-server` | flag | False | Enable HTTP server |
| `--server-port` | int | 8765 | Server port |

## Performance Considerations

1. **Minimal Overhead**: HTTP server runs in separate thread, minimal CPU impact
2. **Event Queue Size**: Default max 1000 events, old events dropped if exceeded
3. **Connection Handling**: `handle_request()` processes one request at a time
4. **Memory**: Each subscriber gets own queue, memory proportional to subscriber count
5. **Network**: SSE uses long-polling pattern, one connection per subscriber

## Future Enhancements

- [ ] WebSocket support for bidirectional communication
- [ ] Event filtering (subscribe to specific event types)
- [ ] Event history replay for late-joining clients
- [ ] Performance metrics endpoint
- [ ] Thumbnail streaming over HTTP
- [ ] Multi-client synchronization
- [ ] Browser-based video player integration

## Files

**New Files:**
- `diveanalyzer/server/__init__.py` - Package marker
- `diveanalyzer/server/sse_server.py` - Main server implementation
- `test_sse_server.py` - Test suite

**Modified Files:**
- `diveanalyzer/cli.py` - CLI integration with server startup/shutdown
- `requirements.txt` - No new dependencies required (uses stdlib only)

## Dependencies

The server uses only Python standard library:
- `http.server` - HTTP server
- `threading` - Background thread
- `queue.Queue` - Thread-safe event queue
- `json` - Event serialization
- `logging` - Event logging
- `pathlib` - File path handling

No additional packages required!

## Troubleshooting

### Port Already in Use
```bash
# Change port with --server-port
diveanalyzer process video.mp4 --enable-server --server-port 9000
```

### Server Won't Start
Check logs for:
- Gallery file missing (run process command with output directory)
- Port already in use (try different port)
- Permission denied (check file permissions)

### No Events Received
- Verify server is running (check `http://localhost:8765/health`)
- Check browser console for JavaScript errors
- Increase `log_level` to "DEBUG" for verbose output

### Connection Refused
- Server may not be started (add `--enable-server` flag)
- Firewall blocking localhost (unlikely but check)
- Port mismatch between server and client

## License

Part of DiveAnalyzer project.
