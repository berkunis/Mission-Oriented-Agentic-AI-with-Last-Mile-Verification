"""
Authentication Middleware
=========================

FastAPI middleware for API authentication.
"""

import sys
from pathlib import Path
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Add security module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from security.auth import APIKeyAuth

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/live",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API authentication.

    Applies API key authentication to all routes except public paths.
    """

    def __init__(self, app, auth_handler: APIKeyAuth | None = None):
        """
        Initialize auth middleware.

        Args:
            app: FastAPI application
            auth_handler: Optional custom auth handler
        """
        super().__init__(app)
        self.auth_handler = auth_handler or APIKeyAuth()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authentication."""
        # Skip auth for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Authenticate
        try:
            api_key = request.headers.get("X-API-Key")
            key_data = await self.auth_handler(request, api_key)
            request.state.api_key_data = key_data
        except Exception as e:
            # Auth handler raises HTTPException
            raise

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (doesn't require auth)."""
        # Exact match
        if path in PUBLIC_PATHS:
            return True

        # Prefix match for docs
        if path.startswith("/docs") or path.startswith("/redoc"):
            return True

        return False
