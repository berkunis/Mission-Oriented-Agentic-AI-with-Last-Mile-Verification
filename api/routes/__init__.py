"""API Routes."""

from api.routes.query import router as query_router
from api.routes.health import router as health_router

__all__ = ["query_router", "health_router"]
