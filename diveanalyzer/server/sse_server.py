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
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict, Any, Callable, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
def get_extract_dive_clip():
    """Import extract_dive_clip on demand."""
    from diveanalyzer.extraction.ffmpeg import extract_dive_clip
    return extract_dive_clip


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
    # Reference to EventServer instance (set by EventServer.start)
    _server_instance = None

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

    def do_POST(self):
        """Handle POST requests for actions like deleting files and extracting selected dives."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Delete files endpoint
        if path == "/delete":
            self._handle_delete_files()
            return

        # Extract selected dives endpoint
        if path == "/extract_selected":
            self._handle_extract_selected()
            return

        # Shutdown endpoint - request server to stop and exit
        if path == "/shutdown":
            self._handle_shutdown()
            return

        # Unknown POST
        self._send_error(404, "Not Found")

    def _handle_extract_selected(self):
        """Handle extraction of selected dives (TICKET-206, TICKET-301).

        Expects JSON body: {"selected_dive_ids": [1, 2, 5, ...]}

        Process:
        1. Parse selected dive IDs from request
        2. Validate against available dives in server.dives_metadata
        3. Start parallel extraction with ThreadPoolExecutor (4 workers)
        4. Emit SSE events for progress tracking
        """
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length > 0 else b''
            payload = json.loads(body.decode('utf-8') or '{}')

            if not isinstance(payload, dict):
                return self._send_error(400, "Invalid payload: must be JSON object")

            selected_dive_ids = payload.get('selected_dive_ids', [])
            if not isinstance(selected_dive_ids, list):
                return self._send_error(400, "Invalid payload: 'selected_dive_ids' must be a list")

            if not selected_dive_ids:
                return self._send_error(400, "No dives selected for extraction")

            # Validate server has required attributes
            if not hasattr(self._server_instance, 'dives_metadata'):
                return self._send_error(500, "Server not properly initialized: no dives metadata")
            if not hasattr(self._server_instance, 'video_path'):
                return self._send_error(500, "Server not properly initialized: no video path")
            if not hasattr(self._server_instance, 'output_dir'):
                return self._send_error(500, "Server not properly initialized: no output directory")

            dives_metadata = self._server_instance.dives_metadata
            video_path = self._server_instance.video_path
            output_dir = self._server_instance.output_dir
            audio_enabled = getattr(self._server_instance, 'audio_enabled', True)

            # Validate all selected IDs exist (dive IDs are 1-indexed based on dive_number)
            available_dive_numbers = {
                dive.dive_number: dive
                for dive in dives_metadata
                if hasattr(dive, 'dive_number')
            }

            invalid_ids = [did for did in selected_dive_ids if did not in available_dive_numbers]
            if invalid_ids:
                return self._send_error(
                    400,
                    f"Invalid dive IDs: {invalid_ids}. Available: {list(available_dive_numbers.keys())}"
                )

            # Filter dives to only selected ones
            selected_dives = [
                available_dive_numbers[did]
                for did in selected_dive_ids
            ]

            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Generate job ID for tracking
            job_id = str(uuid.uuid4())

            # Send acknowledgment response
            response = {
                "status": "started",
                "selected_count": len(selected_dives),
                "extraction_job_id": job_id,
                "message": f"Extracting {len(selected_dives)} selected dives..."
            }
            self._send_json_response(response)

            # Start extraction in background thread to avoid blocking HTTP response
            if self.event_queue:
                extraction_thread = threading.Thread(
                    target=self._extract_selected_dives_background,
                    args=(
                        selected_dives,
                        video_path,
                        output_dir,
                        audio_enabled,
                        job_id,
                        self.event_queue,
                        self._server_instance,  # Pass server instance for gallery regeneration
                    ),
                    daemon=True
                )
                extraction_thread.start()

        except json.JSONDecodeError:
            return self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.exception('Error handling extract_selected request')
            self._send_error(500, f'Error processing extraction request: {e}')

    @staticmethod
    def _extract_selected_dives_background(
        selected_dives,
        video_path: str,
        output_dir: str,
        audio_enabled: bool,
        job_id: str,
        event_queue,
        server_instance=None,
    ):
        """Background thread worker for parallel dive extraction.

        Args:
            selected_dives: List of DiveEvent objects to extract
            video_path: Path to source video
            output_dir: Directory to save extracted dives
            audio_enabled: Whether to include audio
            job_id: Unique job ID for tracking
            event_queue: EventQueue for emitting SSE events
            server_instance: EventServer instance for gallery regeneration
        """
        logger.info(f"🚀 Starting background extraction job {job_id}")

        try:
            extract_dive_clip = get_extract_dive_clip()
            logger.debug(f"✓ extract_dive_clip imported successfully")
        except Exception as e:
            logger.exception(f"❌ Failed to import extract_dive_clip: {e}")
            event_queue.publish('extraction_complete', {
                'job_id': job_id,
                'extracted_count': 0,
                'failed_count': len(selected_dives),
                'total_count': len(selected_dives),
                'failed_dives': [{'dive_id': d.dive_number, 'error': 'import failed'} for d in selected_dives],
                'message': f'Failed to import extraction module: {e}',
            })
            return

        total_count = len(selected_dives)
        extracted_count = 0
        failed_count = 0
        failed_dives = []

        logger.info(f"Preparing to extract {total_count} dives")
        logger.debug(f"Video: {video_path}, Output: {output_dir}, Audio: {audio_enabled}")

        # Emit extraction start event
        event_queue.publish('extraction_started', {
            'job_id': job_id,
            'dive_count': total_count,
            'message': f'Starting extraction of {total_count} dives...'
        })

        # Use ThreadPoolExecutor with 4 worker threads
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}

                # Submit all extraction tasks
                for dive in selected_dives:
                    dive_number = dive.dive_number if hasattr(dive, 'dive_number') else '?'
                    output_filename = f"dive_{dive_number:03d}.mp4"
                    output_path = Path(output_dir) / output_filename

                    logger.debug(f"Submitting extraction task for dive {dive_number}: {dive.start_time}s - {dive.end_time}s")

                    try:
                        future = executor.submit(
                            extract_dive_clip,
                            video_path,
                            dive.start_time,
                            dive.end_time,
                            str(output_path),
                            audio_enabled=audio_enabled,
                        )
                        futures[future] = {
                            'dive': dive,
                            'output_path': output_path,
                            'output_filename': output_filename,
                        }
                        logger.debug(f"✓ Extraction task submitted for dive {dive_number}")
                    except Exception as e:
                        # Handle case where executor is shutting down or task submission fails
                        logger.error(f"Failed to submit extraction task for dive {dive_number}: {type(e).__name__}: {e}")
                        failed_count += 1
                        failed_dives.append({'dive_id': dive_number, 'error': f'submission failed: {e}'})
                        continue

                # Process results as they complete
                for future in futures:
                    dive_info = futures[future]
                    dive = dive_info['dive']
                    output_path = dive_info['output_path']
                    output_filename = dive_info['output_filename']
                    dive_number = dive.dive_number if hasattr(dive, 'dive_number') else '?'

                    try:
                        # Wait for extraction to complete with timeout
                        logger.debug(f"Waiting for extraction of dive {dive_number}...")
                        future.result(timeout=120)  # 2 minute timeout per dive
                        logger.debug(f"✓ Extraction completed for dive {dive_number}")

                        # Check if file was created and get size
                        if output_path.exists():
                            size_mb = output_path.stat().st_size / (1024 * 1024)
                            extracted_count += 1

                            # Emit dive extracted event
                            event_queue.publish('dive_extracted', {
                                'job_id': job_id,
                                'dive_id': dive_number,
                                'success': True,
                                'filename': output_filename,
                                'size_mb': f"{size_mb:.2f}",
                                'extracted_count': extracted_count,
                                'total_count': total_count,
                            })
                            logger.info(f"✅ Extracted dive {dive_number}: {output_filename} ({size_mb:.2f}MB)")
                        else:
                            raise RuntimeError(f"Output file not created: {output_path}")

                    except Exception as e:
                        failed_count += 1
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        failed_dives.append({
                            'dive_id': dive_number,
                            'error': error_msg
                        })

                        # Emit dive extraction failure event
                        event_queue.publish('dive_extracted', {
                            'job_id': job_id,
                            'dive_id': dive_number,
                            'success': False,
                            'error': error_msg,
                            'extracted_count': extracted_count,
                            'total_count': total_count,
                        })
                        logger.error(f"❌ Failed to extract dive {dive_number}: {error_msg}")

        except Exception as e:
            logger.exception(f"Extraction process failed: {e}")

        # Emit extraction complete event
        event_queue.publish('extraction_complete', {
            'job_id': job_id,
            'extracted_count': extracted_count,
            'failed_count': failed_count,
            'total_count': total_count,
            'failed_dives': failed_dives,
            'message': f'Extraction complete: {extracted_count}/{total_count} dives extracted'
                       + (f', {failed_count} failed' if failed_count > 0 else ''),
        })

        logger.info(
            f"Extraction job {job_id} complete: "
            f"{extracted_count}/{total_count} extracted, {failed_count} failed"
        )

        # Regenerate gallery in extracted mode to show video files for review/deletion
        if extracted_count > 0 and server_instance:
            try:
                server_instance.regenerate_gallery_for_extraction_complete()
                logger.info("Gallery regenerated for extracted videos")
            except Exception as e:
                logger.exception(f"Failed to regenerate gallery: {e}")

    def _handle_delete_files(self):
        """Handle deletion of selected dive files.

        Expects JSON body: {"files": ["dive_001.mp4", ...]}
        """
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length > 0 else b''
            payload = json.loads(body.decode('utf-8') or '{}')
            files = payload.get('files', []) if isinstance(payload, dict) else []

            if not isinstance(files, list):
                return self._send_error(400, "Invalid payload: 'files' must be a list")

            base_dir = Path(self.gallery_path).parent if self.gallery_path else Path('.')
            deleted = []
            failed = []

            for fname in files:
                # Sanity: prevent path traversal
                if not isinstance(fname, str) or '..' in fname or fname.startswith('/'):
                    failed.append({"file": fname, "error": "invalid filename"})
                    continue

                target = base_dir / fname
                try:
                    if target.exists() and target.is_file():
                        target.unlink()
                        deleted.append(fname)
                    else:
                        failed.append({"file": fname, "error": "not found"})
                except Exception as e:
                    failed.append({"file": fname, "error": str(e)})

            # Emit event so connected clients can update
            if self.event_queue:
                self.event_queue.publish('files_deleted', {
                    'deleted': deleted,
                    'failed': failed,
                })

            # Return JSON response
            self._send_json_response({
                'deleted': deleted,
                'failed': failed,
                'count': len(deleted),
            })

        except Exception as e:
            logger.exception('Error handling delete files request')
            self._send_error(500, f'Error deleting files: {e}')

    def _handle_shutdown(self):
        """Handle shutdown request from the gallery UI.

        This signals the EventServer to stop running. It does not attempt to
        call `stop()` directly to avoid deadlocks when invoked from the
        server request thread; instead it sets the server's `running` flag
        to False which causes the server loop to exit gracefully.
        """
        try:
            # Notify clients about shutdown
            if self.event_queue:
                self.event_queue.publish('server_shutdown_requested', {
                    'message': 'Shutdown requested by client'
                })

            # If we have an EventServer instance, flip its running flag
            if getattr(DiveReviewSSEHandler, '_server_instance', None):
                try:
                    DiveReviewSSEHandler._server_instance.running = False
                except Exception:
                    # Best-effort; if it fails, continue to respond
                    logger.debug('Could not set server running=False from handler')

            self._send_json_response({'status': 'shutdown_requested'})
            logger.info('Shutdown requested via /shutdown')
        except Exception as e:
            logger.exception('Error handling shutdown request')
            self._send_error(500, f'Error processing shutdown: {e}')

    def _serve_gallery(self):
        """Serve the HTML review gallery file."""
        if not self.gallery_path or not Path(self.gallery_path).exists():
            self._send_error(404, "Gallery not found")
            return

        try:
            # Get file size for Content-Length header
            file_size = Path(self.gallery_path).stat().st_size

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(file_size))
            self.end_headers()

            # Stream file in chunks to handle large files (3.8MB+)
            chunk_size = 65536  # 64KB chunks
            try:
                with open(self.gallery_path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()

                logger.debug(f"Served gallery: {self.gallery_path} ({file_size} bytes)")

            except BrokenPipeError:
                logger.debug(f"Client disconnected while serving gallery")
            except Exception as e:
                logger.error(f"Error streaming gallery: {e}")

        except Exception as e:
            logger.error(f"Error serving gallery: {e}")
            try:
                self._send_error(500, f"Internal Server Error: {e}")
            except:
                # If we can't send error response, just log and return
                logger.error(f"Could not send error response: {e}")

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
            # Use 60 second timeout for keepalive to reduce connection churn
            while True:
                event = self.event_queue.get_event(subscriber_queue, timeout=60.0)

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

        # Attributes for gallery mode switching
        self.video_path = None
        self.output_dir = None
        self.dives_metadata = None
        self.thumbnail_map = None
        self.audio_enabled = True

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
            # Expose server instance to handlers so they can request shutdown
            DiveReviewSSEHandler._server_instance = self

            # Create and configure server with ThreadingHTTPServer for concurrent requests
            # (instead of HTTPServer which blocks on one request at a time)
            self.server = ThreadingHTTPServer((self.host, self.port), DiveReviewSSEHandler)
            self.server.timeout = 1.0  # Timeout for socket operations
            self.server.allow_reuse_address = True

            # Start in background thread (non-daemon so process remains while server runs)
            self.running = True
            self.thread = threading.Thread(target=self._run_server, daemon=False)
            self.thread.start()

            # Give server thread a moment to bind to port before returning
            time.sleep(0.5)

            self.logger.info(f"Server started on http://{self.host}:{self.port}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            self.running = False
            return False

    def _run_server(self):
        """Run the HTTP server (internal thread method).

        Uses ThreadingHTTPServer to handle concurrent requests.
        Continuously polls _BaseServer__shutdown flag for graceful shutdown.
        """
        try:
            self.logger.debug("Server thread started, listening for connections")
            if not self.server:
                self.logger.error("Server not initialized")
                return

            # ThreadingHTTPServer uses daemon threads for request handling
            # We need to periodically check our running flag for graceful shutdown
            self.server.timeout = 0.5  # Short timeout so we check running flag frequently
            while self.running:
                if not self.server:
                    break
                try:
                    # handle_request() processes ONE request and returns
                    # With ThreadingHTTPServer, the request handler runs in its own thread
                    self.server.handle_request()
                except (KeyboardInterrupt, SystemExit):
                    self.logger.info("Received shutdown signal")
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

    def regenerate_gallery_for_extraction_complete(self):
        """Regenerate gallery in extracted mode after dives are extracted.

        This switches the gallery from selection mode (thumbnails) to extracted mode
        (video files for review/deletion). Called after extraction completes.
        """
        try:
            if not self.output_dir:
                logger.warning("Cannot regenerate gallery: output_dir not set")
                return

            from diveanalyzer.utils.review_gallery import DiveGalleryGenerator
            from pathlib import Path

            # Create gallery generator and scan for extracted files
            generator = DiveGalleryGenerator(Path(self.output_dir))
            extracted_dives = generator.scan_output_dir()  # Finds .mp4 files

            if extracted_dives:
                # Regenerate gallery in extracted mode (selection_mode=False)
                generator.selection_mode = False
                new_gallery_path = generator.generate_html()
                self.gallery_path = new_gallery_path

                # Update handler to serve new gallery
                DiveReviewSSEHandler.gallery_path = new_gallery_path

                logger.info(f"Gallery regenerated for extraction: {new_gallery_path}")

                # Emit event to trigger browser reload
                self.emit("gallery_reload_requested", {
                    "message": "Dives extracted! Refreshing gallery for review and deletion...",
                    "extracted_count": len(extracted_dives),
                    "new_gallery_url": self.get_url(),
                })
            else:
                logger.warning("No extracted files found to display")

        except Exception as e:
            logger.exception(f"Error regenerating gallery after extraction: {e}")
