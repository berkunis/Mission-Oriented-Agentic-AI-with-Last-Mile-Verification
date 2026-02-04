"""
FastAPI Application
===================

Main FastAPI application for NL-to-SQL verification service.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api import __version__
from api.middleware.telemetry import TelemetryMiddleware
from api.routes.health import router as health_router
from api.routes.query import router as query_router
from api.schemas import ErrorResponse

# Try to import observability components
try:
    from observability.metrics import setup_metrics, metrics_endpoint
    from observability.tracing import setup_tracing
    from observability.logging_config import setup_logging, get_logger

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


def create_agent():
    """Create and configure the NL-to-SQL agent."""
    from nl_to_sql.agent import NLToSQLAgent
    from nl_to_sql.llm.mock import MockLLM

    # For demo purposes, use MockLLM
    # In production, configure with real LLM
    mock_responses = {
        "customers": ["SELECT * FROM customers"],
        "premium": ["SELECT name, email FROM customers WHERE tier = 'premium'"],
        "orders": ["SELECT * FROM orders"],
        "revenue": ["SELECT SUM(amount) FROM orders"],
        "top": ["SELECT * FROM customers ORDER BY id DESC LIMIT 5"],
    }

    llm = MockLLM(responses=mock_responses)
    return NLToSQLAgent(llm=llm, max_retries=3)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    # Startup
    if OBSERVABILITY_AVAILABLE:
        setup_logging()
        setup_tracing(app)
        logger = get_logger(__name__)
        logger.info("Starting NL-to-SQL API", version=__version__)
    else:
        print(f"Starting NL-to-SQL API v{__version__}")

    # Initialize agent
    app.state.agent = create_agent()

    yield

    # Shutdown
    if OBSERVABILITY_AVAILABLE:
        logger = get_logger(__name__)
        logger.info("Shutting down NL-to-SQL API")
    else:
        print("Shutting down NL-to-SQL API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NL-to-SQL Verified API",
        description=(
            "Mission-oriented agentic AI with last-mile verification. "
            "Converts natural language queries to verified SQL."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add middleware
    app.add_middleware(TelemetryMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routes
    app.include_router(health_router)
    app.include_router(query_router)

    # Add metrics endpoint if available
    if OBSERVABILITY_AVAILABLE:
        setup_metrics(app)
        app.add_route("/metrics", metrics_endpoint)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                request_id=request_id,
            ).model_dump(),
        )

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
