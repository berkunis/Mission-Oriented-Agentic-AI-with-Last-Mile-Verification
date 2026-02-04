"""
Data Models
===========

Core data structures for the NL-to-SQL verification agent.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VerificationStatus(Enum):
    """Status of a verification check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VerificationResult:
    """Result of a single verification step."""

    verifier_name: str
    status: VerificationStatus
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AuditEntry:
    """Single entry in the audit trail."""

    timestamp: str
    step: str
    input_data: dict
    output_data: dict
    verification_results: list[VerificationResult] = field(default_factory=list)


@dataclass
class AgentResult:
    """Final result from the agent."""

    success: bool
    sql: Optional[str]
    original_query: str
    attempts: int
    audit_trail: list[AuditEntry]
    final_message: str


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    tokens_used: int = 0
