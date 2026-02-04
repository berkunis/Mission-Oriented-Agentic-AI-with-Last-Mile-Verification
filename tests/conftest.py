"""
Pytest Fixtures
===============

Shared fixtures for NL-to-SQL agent tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nl_to_sql.agent import NLToSQLAgent
from nl_to_sql.llm.mock import MockLLM
from nl_to_sql.models import VerificationStatus
from nl_to_sql.verifiers.base import VerificationChain
from nl_to_sql.verifiers.safety import SafetyVerifier
from nl_to_sql.verifiers.schema import SchemaVerifier
from nl_to_sql.verifiers.semantic import SemanticVerifier
from nl_to_sql.verifiers.syntax import SAMPLE_SCHEMA, SyntaxVerifier


@pytest.fixture
def sample_schema() -> dict:
    """Return the sample database schema."""
    return SAMPLE_SCHEMA


@pytest.fixture
def syntax_verifier() -> SyntaxVerifier:
    """Create a SyntaxVerifier instance."""
    return SyntaxVerifier()


@pytest.fixture
def schema_verifier() -> SchemaVerifier:
    """Create a SchemaVerifier instance."""
    return SchemaVerifier()


@pytest.fixture
def safety_verifier() -> SafetyVerifier:
    """Create a SafetyVerifier instance."""
    return SafetyVerifier()


@pytest.fixture
def semantic_verifier() -> SemanticVerifier:
    """Create a SemanticVerifier instance."""
    return SemanticVerifier()


@pytest.fixture
def verification_chain() -> VerificationChain:
    """Create a default verification chain."""
    return VerificationChain()


@pytest.fixture
def mock_llm_simple() -> MockLLM:
    """Create a simple mock LLM with basic responses."""
    return MockLLM(
        responses={
            "customers": ["SELECT * FROM customers"],
            "premium": ["SELECT name, email FROM customers WHERE tier = 'premium'"],
            "orders": ["SELECT * FROM orders"],
        }
    )


@pytest.fixture
def mock_llm_with_correction() -> MockLLM:
    """Create a mock LLM that simulates error correction."""
    return MockLLM(
        responses={
            "total revenue": [
                "SELEC SUM(amount) FROM orders",  # Typo: SELEC
                "SELECT SUM(amount) FROM orders",  # Corrected
            ],
            "customer purchases": [
                "SELECT name, total_spent FROM customers",  # Wrong column
                "SELECT c.name, SUM(o.amount) as total_spent FROM customers c "
                "JOIN orders o ON c.id = o.customer_id GROUP BY c.name",
            ],
        }
    )


@pytest.fixture
def agent_simple(mock_llm_simple: MockLLM) -> NLToSQLAgent:
    """Create an agent with simple mock LLM."""
    return NLToSQLAgent(llm=mock_llm_simple, max_retries=2)


@pytest.fixture
def agent_with_correction(mock_llm_with_correction: MockLLM) -> NLToSQLAgent:
    """Create an agent with correction-capable mock LLM."""
    return NLToSQLAgent(llm=mock_llm_with_correction, max_retries=3)


@pytest.fixture
def valid_select_sql() -> str:
    """Return a valid SELECT SQL query."""
    return "SELECT name, email FROM customers WHERE tier = 'premium'"


@pytest.fixture
def invalid_syntax_sql() -> str:
    """Return an invalid SQL query (syntax error)."""
    return "SELEC name FROM customers"


@pytest.fixture
def dangerous_sql() -> str:
    """Return a dangerous SQL query (DELETE without WHERE)."""
    return "DELETE FROM customers"


@pytest.fixture
def verification_context(sample_schema: dict) -> dict:
    """Create a standard verification context."""
    return {
        "schema": sample_schema,
        "original_query": "Show me all premium customers",
    }


def assert_verification_passed(result) -> None:
    """Helper assertion for verification results."""
    assert result.status == VerificationStatus.PASSED, f"Expected PASSED, got: {result.message}"


def assert_verification_failed(result) -> None:
    """Helper assertion for verification failures."""
    assert result.status == VerificationStatus.FAILED, f"Expected FAILED, got: {result.message}"
