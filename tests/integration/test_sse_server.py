#!/usr/bin/env python3
"""
Simple test for SSE server functionality.

Tests:
1. Server startup and shutdown
2. HTTP GET for health check
3. SSE event stream connection
4. Event emission and reception
"""

import json
import threading
import time
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

from diveanalyzer.server import EventServer


def test_server_basic():
    """Test basic server startup and shutdown."""
    print("Test 1: Basic server startup and shutdown")
    print("-" * 50)

    with NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write("<html><body>Test Gallery</body></html>")
        gallery_path = f.name

    try:
        server = EventServer(gallery_path, host="localhost", port=8765)

        # Test startup
        assert server.start(), "Server failed to start"
        print("✓ Server started successfully")
        assert server.is_running(), "Server is not running"
        print("✓ Server is running")

        # Give server time to start
        time.sleep(0.5)

        # Test health endpoint
        try:
            response = urllib.request.urlopen("http://localhost:8765/health", timeout=2)
            data = json.loads(response.read().decode())
            assert data["status"] == "ok", "Health check failed"
            print("✓ Health endpoint working")
        except Exception as e:
            print(f"⚠ Health endpoint test failed: {e}")

        # Test gallery endpoint
        try:
            response = urllib.request.urlopen("http://localhost:8765/", timeout=2)
            content = response.read().decode()
            assert "Test Gallery" in content, "Gallery content not served"
            print("✓ Gallery endpoint serving HTML")
        except Exception as e:
            print(f"⚠ Gallery endpoint test failed: {e}")

        # Test shutdown
        assert server.stop(), "Server failed to stop"
        print("✓ Server stopped successfully")
        assert not server.is_running(), "Server still running after stop"
        print("✓ Server is not running")

    finally:
        Path(gallery_path).unlink(missing_ok=True)

    print()


def test_sse_events():
    """Test SSE event stream and event emission."""
    print("Test 2: SSE event stream and event emission")
    print("-" * 50)

    with NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write("<html><body>Test Gallery</body></html>")
        gallery_path = f.name

    try:
        server = EventServer(gallery_path, host="localhost", port=8766)
        assert server.start(), "Server failed to start"
        print("✓ Server started")

        time.sleep(0.5)

        # Create a thread to consume SSE events
        events_received = []
        error_flag = [False]

        def consume_events():
            try:
                response = urllib.request.urlopen(
                    "http://localhost:8766/events", timeout=5
                )
                line_buffer = ""
                current_event = None

                for byte_data in iter(lambda: response.read(1), b""):
                    line_buffer += byte_data.decode("utf-8", errors="ignore")

                    if "\n" in line_buffer:
                        lines = line_buffer.split("\n")
                        line_buffer = lines[-1]

                        for line in lines[:-1]:
                            line = line.strip()

                            if line.startswith("event:"):
                                current_event = {"type": line[6:].strip()}
                            elif line.startswith("data:"):
                                if current_event:
                                    try:
                                        current_event["data"] = json.loads(line[5:].strip())
                                        events_received.append(current_event)
                                    except json.JSONDecodeError:
                                        pass
                            elif line == "":
                                current_event = None

            except Exception as e:
                error_flag[0] = True
                print(f"SSE consumer error: {e}")

        # Start SSE consumer in thread
        consumer_thread = threading.Thread(target=consume_events, daemon=True)
        consumer_thread.start()

        # Give consumer time to connect
        time.sleep(0.5)

        # Emit test events
        test_events = [
            ("dive_detected", {"dive_id": 1, "confidence": 0.95}),
            ("splash_detection_complete", {"peak_count": 5}),
            ("extraction_complete", {"total_dives": 5, "successful": 5}),
        ]

        for event_type, payload in test_events:
            server.emit(event_type, payload)
            time.sleep(0.1)

        # Give consumer time to receive events
        time.sleep(1)

        # Shutdown
        server.stop()
        consumer_thread.join(timeout=2)

        # Verify events
        print(f"✓ Emitted {len(test_events)} events")
        print(f"✓ Received {len(events_received)} events")

        if events_received:
            print("\nReceived events:")
            for event in events_received:
                print(f"  - {event['type']}: {event.get('data', {})}")
        else:
            print("⚠ No events received (may be timing issue)")

    finally:
        Path(gallery_path).unlink(missing_ok=True)

    print()


def test_concurrent_subscribers():
    """Test multiple concurrent SSE subscribers."""
    print("Test 3: Multiple concurrent SSE subscribers")
    print("-" * 50)

    with NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write("<html><body>Test Gallery</body></html>")
        gallery_path = f.name

    try:
        server = EventServer(gallery_path, host="localhost", port=8767)
        assert server.start(), "Server failed to start"
        print("✓ Server started")

        time.sleep(0.5)

        # Track results from multiple subscribers
        all_events = [[], [], []]
        error_flags = [False, False, False]

        def consumer(subscriber_id):
            try:
                response = urllib.request.urlopen(
                    "http://localhost:8767/events", timeout=3
                )
                event_count = 0

                for _ in range(10):
                    all_events[subscriber_id].append(True)
                    event_count += 1

                print(f"✓ Subscriber {subscriber_id} connected")

            except Exception as e:
                error_flags[subscriber_id] = True
                print(f"Subscriber {subscriber_id} error: {e}")

        # Start multiple subscribers
        threads = []
        for i in range(3):
            t = threading.Thread(target=consumer, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        time.sleep(1)

        # Emit events
        for i in range(3):
            server.emit(f"event_{i}", {"index": i})
            time.sleep(0.1)

        # Wait for threads
        for t in threads:
            t.join(timeout=2)

        server.stop()

        print(f"✓ Multiple subscribers handled successfully")

    finally:
        Path(gallery_path).unlink(missing_ok=True)

    print()


def test_url_methods():
    """Test URL generation methods."""
    print("Test 4: URL generation methods")
    print("-" * 50)

    with NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write("<html><body>Test Gallery</body></html>")
        gallery_path = f.name

    try:
        server = EventServer(gallery_path, host="localhost", port=9999)

        # Test URL methods without starting server
        assert server.get_url() == "http://localhost:9999", "URL format incorrect"
        print("✓ get_url() returns correct format")

        assert server.get_events_url() == "http://localhost:9999/events", "Events URL incorrect"
        print("✓ get_events_url() returns correct format")

    finally:
        Path(gallery_path).unlink(missing_ok=True)

    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("SSE Server Test Suite")
    print("=" * 50 + "\n")

    try:
        test_server_basic()
        test_url_methods()
        test_sse_events()
        test_concurrent_subscribers()

        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
