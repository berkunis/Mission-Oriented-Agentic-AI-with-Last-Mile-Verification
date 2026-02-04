"""
API Schemas
===========

Pydantic models for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for SQL query generation."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language question to convert to SQL",
        examples=["Show me all premium customers"],
    )
    max_retries: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Maximum number of correction attempts (default: 3)",
    )
    include_audit: bool = Field(
        default=False,
        description="Include full audit trail in response",
    )


class VerificationStatusEnum(str, Enum):
    """Verification result status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VerificationResultResponse(BaseModel):
    """Single verification result."""

    verifier_name: str = Field(..., description="Name of the verifier")
    status: VerificationStatusEnum = Field(..., description="Verification status")
    message: str = Field(..., description="Verification message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")


class AuditEntryResponse(BaseModel):
    """Single audit trail entry."""

    timestamp: str = Field(..., description="ISO 8601 timestamp")
    step: str = Field(..., description="Step identifier")
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    verification_results: list[VerificationResultResponse] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response body for SQL query generation."""

    success: bool = Field(..., description="Whether SQL generation succeeded")
    sql: str | None = Field(None, description="Generated SQL query (if successful)")
    original_query: str = Field(..., description="Original natural language query")
    attempts: int = Field(..., description="Number of generation attempts")
    message: str = Field(..., description="Status message")
    audit_trail: list[AuditEntryResponse] | None = Field(
        None,
        description="Full audit trail (if requested)",
    )
    request_id: str = Field(..., description="Unique request identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response."""

    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual component health checks",
    )


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether the service is ready to handle requests")
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual readiness checks",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: str | None = Field(None, description="Request ID if available")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
