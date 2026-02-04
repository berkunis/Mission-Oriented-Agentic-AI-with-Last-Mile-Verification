"""
API Authentication
==================

API key authentication for the NL-to-SQL API.
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# API key header name
API_KEY_HEADER = "X-API-Key"

# In production, store these securely (e.g., database, secrets manager)
# This is a demonstration implementation
_API_KEYS: dict[str, dict] = {}

# Rate limiting storage (in production, use Redis)
_RATE_LIMITS: dict[str, list[datetime]] = {}


class APIKeyAuth:
    """
    API Key authentication handler.

    Supports:
    - API key validation
    - Rate limiting per key
    - Key metadata (scopes, expiration)
    """

    def __init__(
        self,
        rate_limit: int = 100,
        rate_window_seconds: int = 60,
    ):
        """
        Initialize API key authentication.

        Args:
            rate_limit: Maximum requests per window
            rate_window_seconds: Rate limit window in seconds
        """
        self.rate_limit = rate_limit
        self.rate_window = timedelta(seconds=rate_window_seconds)
        self.api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

    async def __call__(
        self,
        request: Request,
        api_key: Annotated[Optional[str], Security(APIKeyHeader(name=API_KEY_HEADER, auto_error=False))] = None,
    ) -> dict:
        """
        Validate API key and check rate limits.

        Returns:
            API key metadata if valid

        Raises:
            HTTPException: If authentication fails
        """
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "AuthenticationRequired",
                    "message": f"Missing {API_KEY_HEADER} header",
                },
            )

        # Validate API key
        key_data = self._validate_key(api_key)
        if key_data is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "InvalidAPIKey",
                    "message": "Invalid or expired API key",
                },
            )

        # Check rate limit
        if not self._check_rate_limit(api_key):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit exceeded. Max {self.rate_limit} requests per {self.rate_window.seconds}s",
                },
                headers={"Retry-After": str(self.rate_window.seconds)},
            )

        # Store key data in request state
        request.state.api_key_data = key_data

        return key_data

    def _validate_key(self, api_key: str) -> Optional[dict]:
        """
        Validate an API key.

        In production, this would check against a database.
        """
        # Hash the key for lookup
        key_hash = self._hash_key(api_key)

        # Check if key exists
        key_data = _API_KEYS.get(key_hash)
        if key_data is None:
            # Check for development key
            if api_key == "dev-key-12345" and os.getenv("ENVIRONMENT") != "production":
                return {
                    "key_id": "dev",
                    "scopes": ["query", "audit"],
                    "rate_limit": 1000,
                }
            return None

        # Check expiration
        if key_data.get("expires_at"):
            if datetime.utcnow() > key_data["expires_at"]:
                return None

        return key_data

    def _check_rate_limit(self, api_key: str) -> bool:
        """Check if request is within rate limits."""
        now = datetime.utcnow()
        window_start = now - self.rate_window

        # Get or create request history
        key_hash = self._hash_key(api_key)
        requests = _RATE_LIMITS.get(key_hash, [])

        # Filter to current window
        requests = [ts for ts in requests if ts > window_start]

        # Check limit
        if len(requests) >= self.rate_limit:
            return False

        # Record this request
        requests.append(now)
        _RATE_LIMITS[key_hash] = requests

        return True

    @staticmethod
    def _hash_key(api_key: str) -> str:
        """Hash an API key for storage/lookup."""
        return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key(
    key_id: str,
    scopes: list[str] = None,
    expires_in_days: int = None,
) -> str:
    """
    Generate a new API key.

    Args:
        key_id: Unique identifier for the key
        scopes: List of allowed scopes (e.g., ["query", "audit"])
        expires_in_days: Days until expiration (None = no expiration)

    Returns:
        The generated API key (store securely - cannot be retrieved later)
    """
    # Generate secure random key
    api_key = f"nlsql_{secrets.token_urlsafe(32)}"

    # Calculate expiration
    expires_at = None
    if expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

    # Store key data (hashed)
    key_hash = APIKeyAuth._hash_key(api_key)
    _API_KEYS[key_hash] = {
        "key_id": key_id,
        "scopes": scopes or ["query"],
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
    }

    return api_key


def revoke_api_key(api_key: str) -> bool:
    """
    Revoke an API key.

    Args:
        api_key: The API key to revoke

    Returns:
        True if key was revoked, False if not found
    """
    key_hash = APIKeyAuth._hash_key(api_key)
    if key_hash in _API_KEYS:
        del _API_KEYS[key_hash]
        return True
    return False


# Dependency for routes
api_key_auth = APIKeyAuth()


async def verify_api_key(
    request: Request,
    api_key: Annotated[Optional[str], Security(APIKeyHeader(name=API_KEY_HEADER, auto_error=False))] = None,
) -> dict:
    """
    FastAPI dependency for API key verification.

    Usage:
        @app.get("/protected")
        async def protected_route(key_data: dict = Depends(verify_api_key)):
            ...
    """
    return await api_key_auth(request, api_key)


def require_scope(required_scope: str):
    """
    Dependency factory to require a specific scope.

    Usage:
        @app.get("/admin")
        async def admin_route(key_data: dict = Depends(require_scope("admin"))):
            ...
    """
    async def check_scope(key_data: dict = Depends(verify_api_key)) -> dict:
        scopes = key_data.get("scopes", [])
        if required_scope not in scopes:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "InsufficientScope",
                    "message": f"This endpoint requires the '{required_scope}' scope",
                },
            )
        return key_data

    return check_scope
