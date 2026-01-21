"""
HTTP Server and SSE (Server-Sent Events) module for live dive review.

Provides real-time event streaming to connected clients for live updates
during video processing and dive detection.
"""

from .sse_server import EventServer

__all__ = ["EventServer"]
