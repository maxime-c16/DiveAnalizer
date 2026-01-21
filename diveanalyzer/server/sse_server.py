"""
HTTP Server with Server-Sent Events (SSE) endpoint for live dive review.

Provides:
- Lightweight HTTP server on configurable host:port
- HTML gallery served at root endpoint
- /events SSE stream for real-time updates
- Background thread operation (non-blocking)
- Graceful shutdown and error handling
- Thread-safe event queue for concurrent connections
"""

import json
import logging
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict, Any, Callable, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class EventQueue:
    """Thread-safe event queue with multiple consumer support and history tracking."""

    def __init__(self, max_size: int = 1000, max_history: int = 1000):
        """Initialize event queue.

        Args:
            max_size: Maximum number of events to keep in queue
            max_history: Maximum number of events to keep in history
        """
        self.queue: Queue = Queue(maxsize=max_size)
        self.subscribers: List[Queue] = []
        self.lock = threading.Lock()
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish an event to all subscribers.

        Args:
            event_type: Type of event (e.g., "dive_detected", "status_update")
            payload: Event payload as dict
        """
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        }

        # Add to main queue
        try:
            self.queue.put_nowait(event)
        except Exception as e:
            logger.warning(f"Could not add event to queue: {e}")

        # Add to history
        with self.lock:
            self.history.append(event)
            if len(self.history) > self.max_history:
                self.history.pop(0)

        # Publish to all subscribers
        with self.lock:
            dead_subscribers = []
            for i, subscriber_queue in enumerate(self.subscribers):
                try:
                    subscriber_queue.put_nowait(event)
                except Exception as e:
                    # Mark dead subscribers for removal
                    dead_subscribers.append(i)
                    logger.debug(f"Could not publish to subscriber {i}: {e}")

            # Remove dead subscribers (iterate in reverse to maintain indices)
            for i in reversed(dead_subscribers):
                self.subscribers.pop(i)

    def subscribe(self) -> Queue:
        """Create a new subscriber queue.

        Returns:
            Queue instance for subscriber to consume events
        """
        subscriber_queue: Queue = Queue()
        with self.lock:
            self.subscribers.append(subscriber_queue)
        logger.debug(f"New subscriber registered. Total subscribers: {len(self.subscribers)}")
        return subscriber_queue

    def get_event(self, subscriber_queue: Queue, timeout: float = 30.0) -> Optional[Dict]:
        """Get next event from subscriber queue with timeout.

        Args:
            subscriber_queue: Queue instance for this subscriber
            timeout: Timeout in seconds to wait for event

        Returns:
            Event dict or None if timeout
        """
        try:
            return subscriber_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history.

        Args:
            limit: Maximum number of recent events to return

        Returns:
            List of recent events, limited to specified count
        """
        with self.lock:
            # Return the most recent events (up to limit)
            if limit <= 0:
                return list(self.history)
            return list(self.history[-limit:])


class DiveReviewSSEHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dive review server."""

    # Class variables shared across instances
    gallery_path: Optional[str] = None
    event_queue: Optional[EventQueue] = None

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Serve review gallery at root
        if path == "/" or path == "":
            self._serve_gallery()

        # SSE events endpoint
        elif path == "/events":
            self._handle_sse_stream()

        # Event history endpoint (FEAT-08: Connection Management)
        elif path == "/events-history":
            self._handle_events_history()

        # Health check endpoint
        elif path == "/health":
            self._send_json_response({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

        # 404
        else:
            self._send_error(404, "Not Found")

    def _serve_gallery(self):
        """Serve the HTML review gallery file."""
        if not self.gallery_path or not Path(self.gallery_path).exists():
            self._send_error(404, "Gallery not found")
            return

        try:
            with open(self.gallery_path, "r") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))

            logger.debug(f"Served gallery: {self.gallery_path}")

        except Exception as e:
            logger.error(f"Error serving gallery: {e}")
            self._send_error(500, f"Internal Server Error: {e}")

    def _handle_events_history(self):
        """Handle event history requests (FEAT-08: Polling fallback)."""
        if not self.event_queue:
            self._send_error(500, "Event queue not initialized")
            return

        try:
            # Get history (last 100 events)
            history = self.event_queue.get_history(limit=100)

            response = {
                "events": history,
                "count": len(history),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            self._send_json_response(response)
            logger.debug(f"Served {len(history)} events from history")

        except Exception as e:
            logger.error(f"Error serving event history: {e}")
            self._send_error(500, f"Error retrieving history: {e}")

    def _handle_sse_stream(self):
        """Handle Server-Sent Events stream for live updates."""
        if not self.event_queue:
            self._send_error(500, "Event queue not initialized")
            return

        try:
            # Get subscriber queue
            subscriber_queue = self.event_queue.subscribe()

            # Send SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            logger.info(
                f"SSE client connected from {self.client_address[0]}:{self.client_address[1]}"
            )

            # Send initial connection event
            self._send_sse_event("connected", {"message": "SSE stream established"})

            # Stream events indefinitely
            while True:
                event = self.event_queue.get_event(subscriber_queue, timeout=30.0)

                if event:
                    # Send event to client
                    self._send_sse_event(event["event_type"], event["payload"])
                else:
                    # Send keepalive comment
                    self._send_sse_keepalive()

        except BrokenPipeError:
            logger.debug(f"SSE client disconnected: {self.client_address}")
        except Exception as e:
            logger.error(f"SSE stream error: {e}")

    def _send_sse_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Send SSE formatted event to client.

        Args:
            event_type: Type of event
            data: Event data payload

        Returns:
            True if sent successfully, False if error
        """
        try:
            event_json = json.dumps(data)
            sse_message = f"event: {event_type}\ndata: {event_json}\n\n"
            self.wfile.write(sse_message.encode("utf-8"))
            self.wfile.flush()
            return True
        except Exception as e:
            logger.debug(f"Error sending SSE event: {e}")
            return False

    def _send_sse_keepalive(self) -> bool:
        """Send SSE keepalive comment.

        Returns:
            True if sent successfully, False if error
        """
        try:
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
            return True
        except Exception as e:
            logger.debug(f"Error sending keepalive: {e}")
            return False

    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response.

        Args:
            data: Response data
            status_code: HTTP status code
        """
        response = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _send_error(self, status_code: int, message: str):
        """Send error response.

        Args:
            status_code: HTTP status code
            message: Error message
        """
        self._send_json_response(
            {"error": message, "status_code": status_code}, status_code=status_code
        )

    def log_message(self, format, *args):
        """Override to use logger instead of stderr."""
        logger.debug(f"{self.client_address[0]} - {format % args}")


class EventServer:
    """HTTP server with SSE endpoint for live dive review.

    Runs in background thread, serving HTML gallery and providing real-time
    event stream for live updates during video processing.

    Example:
        server = EventServer(
            gallery_path="./dives/review_gallery.html",
            host="localhost",
            port=8765
        )
        server.start()

        # Emit events during processing
        server.emit("dive_detected", {"dive_id": 1, "confidence": 0.95})
        server.emit("thumbnail_ready", {"dive_id": 1, "path": "dive_1.jpg"})

        # Cleanup
        server.stop()
    """

    def __init__(
        self,
        gallery_path: str,
        host: str = "localhost",
        port: int = 8765,
        log_level: str = "INFO",
    ):
        """Initialize event server.

        Args:
            gallery_path: Path to review gallery HTML file
            host: Host to bind to (default: localhost)
            port: Port to listen on (default: 8765)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.gallery_path = gallery_path
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

        # Shared state for request handlers
        self.event_queue = EventQueue()

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def start(self) -> bool:
        """Start the HTTP server in a background thread.

        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            self.logger.warning("Server already running")
            return False

        try:
            # Verify gallery exists
            if not Path(self.gallery_path).exists():
                self.logger.error(f"Gallery file not found: {self.gallery_path}")
                return False

            # Set class variables for request handler
            DiveReviewSSEHandler.gallery_path = self.gallery_path
            DiveReviewSSEHandler.event_queue = self.event_queue

            # Create and configure server
            self.server = HTTPServer((self.host, self.port), DiveReviewSSEHandler)
            self.server.timeout = 1.0  # Timeout for socket operations
            self.server.allow_reuse_address = True

            # Start in background thread
            self.running = True
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()

            self.logger.info(f"Server started on http://{self.host}:{self.port}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            self.running = False
            return False

    def _run_server(self):
        """Run the HTTP server (internal thread method)."""
        try:
            self.logger.debug("Server thread started, listening for connections")
            while self.running:
                if not self.server:
                    break
                try:
                    # Use handle_request to process one request at a time
                    # This allows graceful shutdown
                    self.server.handle_request()
                except (KeyboardInterrupt, SystemExit):
                    break
                except Exception as e:
                    if self.running:
                        self.logger.debug(f"Request handling error: {e}")
        except Exception as e:
            if self.running:
                self.logger.error(f"Server error: {e}")
        finally:
            self.logger.debug("Server thread shutting down")
            if self.server:
                try:
                    self.server.server_close()
                except Exception as e:
                    self.logger.debug(f"Error closing server: {e}")
            self.logger.debug("Server thread ended")

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit an event to all connected SSE clients.

        Args:
            event_type: Type of event (e.g., "dive_detected", "status_update")
            payload: Event payload as dict

        Example:
            server.emit("dive_detected", {
                "dive_id": 1,
                "confidence": 0.95,
                "duration": 1.2
            })
        """
        if not self.running:
            self.logger.warning("Cannot emit event: server not running")
            return

        self.event_queue.publish(event_type, payload)
        self.logger.debug(f"Event emitted: {event_type}")

    def stop(self) -> bool:
        """Stop the server and wait for thread cleanup.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            return True

        try:
            self.running = False

            # Close server socket
            if self.server:
                self.server.server_close()

            # Wait for thread to finish (with timeout)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)

                if self.thread.is_alive():
                    self.logger.warning("Server thread did not stop within timeout")
                    return False

            self.logger.info("Server stopped cleanly")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")
            return False

    def get_url(self) -> str:
        """Get the server URL.

        Returns:
            Full URL to server (e.g., http://localhost:8765)
        """
        return f"http://{self.host}:{self.port}"

    def get_events_url(self) -> str:
        """Get the SSE events endpoint URL.

        Returns:
            Full URL to events endpoint (e.g., http://localhost:8765/events)
        """
        return f"{self.get_url()}/events"

    def is_running(self) -> bool:
        """Check if server is running.

        Returns:
            True if server is currently running
        """
        return self.running
