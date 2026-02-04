"""
Health Check Routes
===================

Kubernetes-compatible health and readiness endpoints.
"""

from datetime import datetime

from fastapi import APIRouter

from api import __version__
from api.schemas import HealthResponse, HealthStatus, ReadinessResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the service",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        HealthResponse with current service status
    """
    # Perform health checks
    checks = {
        "api": True,
        "verifiers": True,  # Could add actual verifier health checks
    }

    # Determine overall status
    all_healthy = all(checks.values())
    status = HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED

    return HealthResponse(
        status=status,
        version=__version__,
        timestamp=datetime.utcnow(),
        checks=checks,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Returns whether the service is ready to handle requests",
)
async def readiness_check() -> ReadinessResponse:
    """
    Readiness check for Kubernetes.

    Used to determine if the pod should receive traffic.

    Returns:
        ReadinessResponse indicating readiness status
    """
    checks = {
        "dependencies_loaded": True,
        "configuration_valid": True,
    }

    return ReadinessResponse(
        ready=all(checks.values()),
        checks=checks,
    )


@router.get(
    "/live",
    summary="Liveness check",
    description="Simple liveness probe",
)
async def liveness_check() -> dict:
    """
    Liveness check for Kubernetes.

    Simple endpoint to verify the process is running.

    Returns:
        Simple OK response
    """
    return {"status": "ok"}
