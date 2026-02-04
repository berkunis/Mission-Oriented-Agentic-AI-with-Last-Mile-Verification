"""
NL-to-SQL Verified Agent
========================

Mission-oriented agentic AI with last-mile verification for NL-to-SQL conversion.
"""

from nl_to_sql.models import (
    AgentResult,
    AuditEntry,
    LLMResponse,
    VerificationResult,
    VerificationStatus,
)
from nl_to_sql.agent import NLToSQLAgent, SAMPLE_SCHEMA
from nl_to_sql.verifiers import (
    Verifier,
    VerificationChain,
    SyntaxVerifier,
    SchemaVerifier,
    SafetyVerifier,
    SemanticVerifier,
)
from nl_to_sql.llm import LLMInterface, MockLLM

__version__ = "0.1.0"

__all__ = [
    # Models
    "VerificationStatus",
    "VerificationResult",
    "AuditEntry",
    "AgentResult",
    "LLMResponse",
    # Agent
    "NLToSQLAgent",
    "SAMPLE_SCHEMA",
    # Verifiers
    "Verifier",
    "VerificationChain",
    "SyntaxVerifier",
    "SchemaVerifier",
    "SafetyVerifier",
    "SemanticVerifier",
    # LLM
    "LLMInterface",
    "MockLLM",
]
