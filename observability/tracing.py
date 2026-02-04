"""
OpenTelemetry Tracing
=====================

Distributed tracing for request flow visualization.
"""

import os
from typing import Optional

from fastapi import FastAPI

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


def setup_tracing(
    app: FastAPI,
    service_name: str = "nl-to-sql-api",
    otlp_endpoint: Optional[str] = None,
) -> None:
    """
    Set up OpenTelemetry tracing for the application.

    Args:
        app: FastAPI application instance
        service_name: Name of the service for traces
        otlp_endpoint: OTLP collector endpoint (default: from env or localhost:4317)
    """
    if not OTEL_AVAILABLE:
        print("OpenTelemetry not available. Tracing disabled.")
        return

    # Get endpoint from environment or parameter
    endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # Create resource with service info
    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.version": "0.1.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint is configured
    if endpoint and endpoint != "disabled":
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except Exception as e:
            print(f"Failed to configure OTLP exporter: {e}")

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str = __name__):
    """
    Get a tracer instance for creating spans.

    Args:
        name: Name for the tracer (usually module name)

    Returns:
        Tracer instance or NoOp tracer if not available
    """
    if not OTEL_AVAILABLE:
        return NoOpTracer()

    return trace.get_tracer(name)


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value) -> None:
        pass

    def add_event(self, name: str, attributes: dict = None) -> None:
        pass

    def set_status(self, status) -> None:
        pass


class NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()


def trace_agent_operation(operation_name: str):
    """
    Decorator to trace agent operations.

    Args:
        operation_name: Name of the operation for the span
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer(__name__)
            with tracer.start_as_current_span(operation_name) as span:
                try:
                    result = func(*args, **kwargs)
                    if hasattr(result, 'success'):
                        span.set_attribute("query.success", result.success)
                        span.set_attribute("query.attempts", result.attempts)
                    return result
                except Exception as e:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    raise
        return wrapper
    return decorator
