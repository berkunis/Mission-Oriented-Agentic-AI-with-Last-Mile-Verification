"""
Telemetry Middleware
====================

Request/response telemetry and correlation ID handling.
"""

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request telemetry and correlation ID propagation.

    Adds:
    - X-Request-ID header to all responses
    - Request timing
    - Logging context
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with telemetry."""
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state for access in routes
        request.state.request_id = request_id

        # Record start time
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add telemetry headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        return response
