# HTTP Server Quick Start

## Enable the Server

```bash
# Basic usage (default port 8765)
diveanalyzer process video.mp4 --enable-server

# Custom port
diveanalyzer process video.mp4 --enable-server --server-port 9000

# Verbose logging
diveanalyzer process video.mp4 --enable-server -v
```

## Access the Server

Once running, the server is available at:
- **Gallery**: http://localhost:8765
- **Events**: http://localhost:8765/events
- **Health**: http://localhost:8765/health

## Browser Client Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dive Processing Monitor</title>
</head>
<body>
    <h1>Dive Processing Monitor</h1>
    <div id="status">Connecting...</div>
    <div id="events"></div>

    <script>
        const eventDiv = document.getElementById('events');
        const statusDiv = document.getElementById('status');

        const eventSource = new EventSource('http://localhost:8765/events');

        eventSource.onopen = () => {
            statusDiv.textContent = 'Connected ✓';
        };

        eventSource.addEventListener('dives_detected', (e) => {
            const data = JSON.parse(e.data);
            const msg = `<p>🎬 Detected ${data.dive_count} dives (${data.signal_type})</p>`;
            eventDiv.innerHTML += msg;
        });

        eventSource.addEventListener('extraction_complete', (e) => {
            const data = JSON.parse(e.data);
            const msg = `<p>✂️  Extracted ${data.successful}/${data.total_dives} clips</p>`;
            eventDiv.innerHTML += msg;
        });

        eventSource.addEventListener('processing_complete', (e) => {
            const data = JSON.parse(e.data);
            const msg = `<p>✅ Processing complete - Output: ${data.output_directory}</p>`;
            eventDiv.innerHTML += msg;
        });

        eventSource.onerror = () => {
            statusDiv.textContent = 'Disconnected ✗';
        };
    </script>
</body>
</html>
```

## Python Client Example

```python
import json
import requests

url = 'http://localhost:8765/events'
response = requests.get(url, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('event:'):
            event_type = line[6:].strip()
            print(f"Event: {event_type}")
        elif line.startswith('data:'):
            data = json.loads(line[5:].strip())
            print(f"Data: {data}")
```

## Event Flow During Processing

```
1. Server starts → Listening on http://localhost:8765
   └─ Clients can connect to /events endpoint

2. Audio extraction
   └─ splash_detection_complete event

3. Motion validation (if enabled)
   └─ motion_detection_complete event

4. Person detection (if enabled)
   └─ person_detection_complete event

5. Signal fusion
   └─ dives_detected event

6. Clip extraction
   └─ extraction_complete event

7. Processing done
   └─ processing_complete event
   └─ Server shuts down gracefully
```

## Event Format

All events follow this JSON structure:

```json
{
  "event_type": "dives_detected",
  "timestamp": "2026-01-21T12:34:56.789Z",
  "payload": {
    "dive_count": 5,
    "signal_type": "audio + motion",
    "confidence_threshold": 0.5
  }
}
```

## Troubleshooting

### Port already in use
```bash
diveanalyzer process video.mp4 --enable-server --server-port 9000
```

### Check server health
```bash
curl http://localhost:8765/health
# {"status": "ok", "timestamp": "..."}
```

### View server logs
```bash
diveanalyzer process video.mp4 --enable-server -v
```

### Verify gallery is served
```bash
curl http://localhost:8765 | head -20
```

## Complete Example

```bash
#!/bin/bash

# Start processing with server in background
diveanalyzer process session.mov --enable-server --server-port 8765 &

# Give server time to start
sleep 1

# Monitor events in another terminal
curl -N http://localhost:8765/events | while read line; do
    echo "[$(date '+%H:%M:%S')] $line"
done
```

## Architecture Highlights

✓ **Non-blocking**: Server runs in background thread, doesn't slow down processing
✓ **Thread-safe**: Event queue properly synchronized for multiple subscribers
✓ **Minimal overhead**: Uses Python stdlib only, <500ms startup time
✓ **Graceful shutdown**: Proper cleanup on completion, all sockets closed
✓ **Error handling**: Handles client disconnects, network issues gracefully
✓ **Logging**: Full debug logging available with `-v` flag

## Next Steps

- [ ] Integrate with frontend dashboard
- [ ] Add WebSocket support for bi-directional communication
- [ ] Implement event replay for late-joining clients
- [ ] Add thumbnail streaming endpoint
- [ ] Multi-client synchronization
