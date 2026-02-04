"""
Structured Logging Configuration
================================

JSON-structured logging with context propagation.
"""

import logging
import os
import sys
from typing import Any

try:
    import structlog
    from structlog.types import Processor

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


def setup_logging(
    level: str = None,
    json_format: bool = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (default: from LOG_LEVEL env or INFO)
        json_format: Whether to use JSON format (default: from LOG_FORMAT env or True in production)
    """
    # Get configuration from environment
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    environment = os.getenv("ENVIRONMENT", "development")

    # Use JSON in production, pretty print in development
    if json_format is None:
        json_format = os.getenv("LOG_FORMAT", "").lower() == "json" or environment == "production"

    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
        )
        return

    # Configure structlog processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON format for production
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty print for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to use structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str = None) -> Any:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger instance
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)


def bind_context(**kwargs) -> None:
    """
    Bind context variables to all subsequent log messages.

    Args:
        **kwargs: Context key-value pairs to bind
    """
    if STRUCTLOG_AVAILABLE:
        structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    if STRUCTLOG_AVAILABLE:
        structlog.contextvars.clear_contextvars()


class LoggerAdapter:
    """
    Adapter to provide consistent logging interface.

    Works with both structlog and standard logging.
    """

    def __init__(self, name: str = None):
        self.logger = get_logger(name)

    def info(self, message: str, **kwargs) -> None:
        self.logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        self.logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.logger.error(message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        self.logger.exception(message, **kwargs)

    def bind(self, **kwargs) -> "LoggerAdapter":
        """Bind context and return self for chaining."""
        if STRUCTLOG_AVAILABLE:
            self.logger = self.logger.bind(**kwargs)
        return self
