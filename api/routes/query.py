"""
Query Routes
============

Main API endpoint for NL-to-SQL conversion.
"""

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import (
    AuditEntryResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    VerificationResultResponse,
    VerificationStatusEnum,
)

router = APIRouter(prefix="/api/v1", tags=["Query"])


def get_agent(request: Request):
    """Dependency to get the configured agent from app state."""
    return request.app.state.agent


def get_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Convert natural language to verified SQL",
    description="Takes a natural language question and returns a verified SQL query",
)
async def process_query(
    request: QueryRequest,
    agent=Depends(get_agent),
    request_id: Annotated[str, Depends(get_request_id)] = None,
) -> QueryResponse:
    """
    Process a natural language query and return verified SQL.

    The endpoint:
    1. Accepts a natural language question
    2. Generates SQL using the configured LLM
    3. Verifies the SQL through the verification chain
    4. Returns the verified SQL or error details

    Args:
        request: Query request with natural language question
        agent: Injected NLToSQLAgent instance
        request_id: Auto-generated unique request ID

    Returns:
        QueryResponse with verified SQL and metadata
    """
    start_time = time.perf_counter()

    try:
        # Override max_retries if specified
        original_max_retries = agent.max_retries
        if request.max_retries is not None:
            agent.max_retries = request.max_retries

        # Process the query
        result = agent.process(request.query)

        # Restore original max_retries
        agent.max_retries = original_max_retries

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Convert audit trail if requested
        audit_trail = None
        if request.include_audit:
            audit_trail = [
                AuditEntryResponse(
                    timestamp=entry.timestamp,
                    step=entry.step,
                    input_data=entry.input_data,
                    output_data=entry.output_data,
                    verification_results=[
                        VerificationResultResponse(
                            verifier_name=vr.verifier_name,
                            status=VerificationStatusEnum(vr.status.value),
                            message=vr.message,
                            details=vr.details,
                        )
                        for vr in entry.verification_results
                    ],
                )
                for entry in result.audit_trail
            ]

        return QueryResponse(
            success=result.success,
            sql=result.sql,
            original_query=result.original_query,
            attempts=result.attempts,
            message=result.final_message,
            audit_trail=audit_trail,
            request_id=request_id,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ProcessingError",
                "message": str(e),
                "request_id": request_id,
            },
        )
