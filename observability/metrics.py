"""
Prometheus Metrics
==================

Application metrics for monitoring and alerting.
"""

import time
from functools import wraps
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
)

# Create a custom registry for this application
REGISTRY = CollectorRegistry()

# Application info
APP_INFO = Info(
    "nl_to_sql",
    "NL-to-SQL application information",
    registry=REGISTRY,
)

# Query metrics
QUERIES_TOTAL = Counter(
    "nl_to_sql_queries_total",
    "Total number of NL-to-SQL queries processed",
    ["status"],  # success, failure
    registry=REGISTRY,
)

QUERY_DURATION = Histogram(
    "nl_to_sql_query_duration_seconds",
    "Query processing duration in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

QUERY_ATTEMPTS = Histogram(
    "nl_to_sql_query_attempts",
    "Number of attempts per query",
    buckets=[1, 2, 3, 4, 5],
    registry=REGISTRY,
)

# Verification metrics
VERIFICATION_FAILURES = Counter(
    "nl_to_sql_verification_failures_total",
    "Total verification failures by verifier",
    ["verifier"],
    registry=REGISTRY,
)

VERIFICATION_DURATION = Histogram(
    "nl_to_sql_verification_duration_seconds",
    "Verification chain duration in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY,
)

# HTTP metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

# Active queries gauge
ACTIVE_QUERIES = Gauge(
    "nl_to_sql_active_queries",
    "Number of queries currently being processed",
    registry=REGISTRY,
)


def setup_metrics(app: FastAPI) -> None:
    """
    Set up Prometheus metrics for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Set application info
    APP_INFO.info({
        "version": "0.1.0",
        "environment": "development",
    })

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        """Middleware to track HTTP metrics."""
        start_time = time.perf_counter()

        # Track active queries for query endpoint
        is_query_endpoint = request.url.path == "/api/v1/query"
        if is_query_endpoint:
            ACTIVE_QUERIES.inc()

        try:
            response = await call_next(request)
            duration = time.perf_counter() - start_time

            # Record metrics
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).inc()

            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)

            return response
        finally:
            if is_query_endpoint:
                ACTIVE_QUERIES.dec()


def track_query_metrics(
    success: bool,
    attempts: int,
    duration_seconds: float,
    failed_verifiers: list[str] | None = None,
) -> None:
    """
    Track metrics for a completed query.

    Args:
        success: Whether the query succeeded
        attempts: Number of attempts made
        duration_seconds: Total processing time
        failed_verifiers: List of verifier names that failed
    """
    # Track overall query metrics
    QUERIES_TOTAL.labels(status="success" if success else "failure").inc()
    QUERY_DURATION.observe(duration_seconds)
    QUERY_ATTEMPTS.observe(attempts)

    # Track verification failures
    if failed_verifiers:
        for verifier in failed_verifiers:
            VERIFICATION_FAILURES.labels(verifier=verifier).inc()


async def metrics_endpoint(request: Request) -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    # Handle multiprocess mode if using gunicorn
    try:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        metrics = generate_latest(registry)
    except ValueError:
        # Not in multiprocess mode
        metrics = generate_latest(REGISTRY)

    return Response(
        content=metrics,
        media_type=CONTENT_TYPE_LATEST,
    )


def timed_operation(metric: Histogram):
    """
    Decorator to time operations and record to histogram.

    Args:
        metric: Prometheus Histogram to record to
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start_time
                metric.observe(duration)
        return wrapper
    return decorator
