"""
Unit Tests for NLToSQLAgent
===========================

Tests for the main agent controller.
"""

import pytest

from nl_to_sql.agent import NLToSQLAgent
from nl_to_sql.llm.mock import MockLLM
from nl_to_sql.models import VerificationStatus


class TestAgentBasic:
    """Basic agent functionality tests."""

    def test_agent_creation(self, mock_llm_simple: MockLLM) -> None:
        """Test that agent can be created with default settings."""
        agent = NLToSQLAgent(llm=mock_llm_simple)
        assert agent.max_retries == 3
        assert agent.verification_chain is not None
        assert agent.schema is not None

    def test_agent_custom_retries(self, mock_llm_simple: MockLLM) -> None:
        """Test that custom retry count is respected."""
        agent = NLToSQLAgent(llm=mock_llm_simple, max_retries=5)
        assert agent.max_retries == 5

    def test_simple_query_success(self, agent_simple: NLToSQLAgent) -> None:
        """Test that a simple valid query succeeds."""
        result = agent_simple.process("Show me all customers")
        assert result.success is True
        assert result.sql is not None
        assert "customers" in result.sql.lower()
        assert result.attempts == 1

    def test_query_returns_audit_trail(self, agent_simple: NLToSQLAgent) -> None:
        """Test that audit trail is populated."""
        result = agent_simple.process("Show me all customers")
        assert len(result.audit_trail) > 0
        # Should have generation, response, and verification entries
        steps = [entry.step for entry in result.audit_trail]
        assert any("generation" in s for s in steps)
        assert any("llm_response" in s for s in steps)
        assert any("verification" in s for s in steps)

    def test_audit_trail_reset_per_query(self, agent_simple: NLToSQLAgent) -> None:
        """Test that audit trail resets between queries."""
        result1 = agent_simple.process("Show me all customers")
        trail_len_1 = len(result1.audit_trail)

        result2 = agent_simple.process("Show me premium customers")
        trail_len_2 = len(result2.audit_trail)

        # Trails should be similar length (both simple queries)
        assert trail_len_1 == trail_len_2


class TestAgentCorrection:
    """Tests for agent self-correction behavior."""

    def test_syntax_error_correction(self) -> None:
        """Test that syntax errors trigger correction."""
        llm = MockLLM(
            responses={
                "revenue": [
                    "SELEC SUM(amount) FROM orders",  # Typo
                    "SELECT SUM(amount) FROM orders",  # Fixed
                ]
            }
        )
        agent = NLToSQLAgent(llm=llm, max_retries=2)
        result = agent.process("What is the total revenue?")

        assert result.success is True
        assert result.attempts == 2  # Needed correction

    def test_schema_error_correction(self) -> None:
        """Test that schema errors trigger correction."""
        llm = MockLLM(
            responses={
                "purchases": [
                    "SELECT total_spent FROM customers",  # Invalid column
                    "SELECT name FROM customers",  # Valid
                ]
            }
        )
        agent = NLToSQLAgent(llm=llm, max_retries=2)
        result = agent.process("Show customer purchases")

        assert result.success is True
        assert result.attempts == 2

    def test_max_retries_exceeded(self) -> None:
        """Test that max retries is respected."""
        llm = MockLLM(
            responses={
                "bad": [
                    "INVALID SQL",  # Always bad
                ]
            }
        )
        agent = NLToSQLAgent(llm=llm, max_retries=2)
        result = agent.process("bad query")

        assert result.success is False
        assert result.attempts == 3  # Initial + 2 retries
        assert "Failed" in result.final_message

    def test_correction_audit_trail(self) -> None:
        """Test that corrections are recorded in audit trail."""
        llm = MockLLM(
            responses={
                "revenue": [
                    "SELEC SUM(amount) FROM orders",  # Typo
                    "SELECT SUM(amount) FROM orders",  # Fixed
                ]
            }
        )
        agent = NLToSQLAgent(llm=llm, max_retries=2)
        result = agent.process("What is the total revenue?")

        # Should have entries for both attempts
        steps = [entry.step for entry in result.audit_trail]
        assert "generation_attempt_1" in steps
        assert "generation_attempt_2" in steps
        assert "verification_1" in steps
        assert "verification_2" in steps

        # First verification should have failed
        verification_1 = next(
            e for e in result.audit_trail if e.step == "verification_1"
        )
        failed_results = [
            r for r in verification_1.verification_results
            if r.status == VerificationStatus.FAILED
        ]
        assert len(failed_results) > 0


class TestAgentSQLExtraction:
    """Tests for SQL extraction from LLM output."""

    def test_plain_sql(self) -> None:
        """Test extraction of plain SQL."""
        llm = MockLLM(responses={"test": ["SELECT * FROM customers"]})
        agent = NLToSQLAgent(llm=llm)
        result = agent.process("test query")

        assert result.sql == "SELECT * FROM customers"

    def test_markdown_code_block(self) -> None:
        """Test extraction from markdown code block."""
        llm = MockLLM(
            responses={"test": ["```sql\nSELECT * FROM customers\n```"]}
        )
        agent = NLToSQLAgent(llm=llm)
        result = agent.process("test query")

        assert result.sql == "SELECT * FROM customers"

    def test_whitespace_trimming(self) -> None:
        """Test that whitespace is trimmed."""
        llm = MockLLM(
            responses={"test": ["  \n  SELECT * FROM customers  \n  "]}
        )
        agent = NLToSQLAgent(llm=llm)
        result = agent.process("test query")

        assert result.sql == "SELECT * FROM customers"


class TestAgentVerificationContext:
    """Tests for verification context handling."""

    def test_original_query_in_context(self) -> None:
        """Test that original query is passed to verifiers."""
        llm = MockLLM(
            responses={
                "top": [
                    "SELECT name FROM customers ORDER BY id DESC LIMIT 5"
                ]
            }
        )
        agent = NLToSQLAgent(llm=llm)
        result = agent.process("Show top 5 customers")

        assert result.success is True
        # Semantic verifier would fail without LIMIT if context wasn't passed

    def test_custom_schema(self) -> None:
        """Test that custom schema is used."""
        custom_schema = {
            "employees": {
                "columns": ["id", "name", "department"],
                "types": {"id": "INTEGER", "name": "TEXT", "department": "TEXT"},
            }
        }
        llm = MockLLM(responses={"employees": ["SELECT * FROM employees"]})
        agent = NLToSQLAgent(llm=llm, schema=custom_schema)
        result = agent.process("Show employees")

        assert result.success is True


class TestMockLLM:
    """Tests for the MockLLM implementation."""

    def test_response_matching(self) -> None:
        """Test that responses are matched by keyword."""
        llm = MockLLM(
            responses={
                "customer": ["SELECT * FROM customers"],
                "order": ["SELECT * FROM orders"],
            }
        )

        response1 = llm.generate("Show me customers")
        assert "customers" in response1.content

        response2 = llm.generate("Show me orders")
        assert "orders" in response2.content

    def test_sequential_responses(self) -> None:
        """Test that responses cycle through attempts."""
        llm = MockLLM(
            responses={
                "test": ["first", "second", "third"],
            }
        )

        assert llm.generate("test").content == "first"
        assert llm.generate("test").content == "second"
        assert llm.generate("test").content == "third"
        assert llm.generate("test").content == "third"  # Stays at last

    def test_reset(self) -> None:
        """Test that reset clears call counts."""
        llm = MockLLM(responses={"test": ["first", "second"]})

        llm.generate("test")
        llm.generate("test")
        llm.reset()
        assert llm.generate("test").content == "first"

    def test_default_fallback(self) -> None:
        """Test that unknown queries get fallback response."""
        llm = MockLLM(responses={})
        response = llm.generate("unknown query")
        assert "unknown_table" in response.content
