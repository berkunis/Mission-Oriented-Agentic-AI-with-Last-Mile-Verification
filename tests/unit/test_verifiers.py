"""
Unit Tests for Verifiers
========================

Tests for each verifier in the verification chain.
"""

import pytest

from nl_to_sql.models import VerificationStatus
from nl_to_sql.verifiers.safety import SafetyVerifier
from nl_to_sql.verifiers.schema import SchemaVerifier
from nl_to_sql.verifiers.semantic import SemanticVerifier
from nl_to_sql.verifiers.syntax import SyntaxVerifier


class TestSyntaxVerifier:
    """Tests for the SyntaxVerifier."""

    def test_valid_select(
        self, syntax_verifier: SyntaxVerifier, verification_context: dict
    ) -> None:
        """Test that valid SELECT passes verification."""
        sql = "SELECT name, email FROM customers WHERE tier = 'premium'"
        result = syntax_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED
        assert result.verifier_name == "SyntaxVerifier"

    def test_valid_join(
        self, syntax_verifier: SyntaxVerifier, verification_context: dict
    ) -> None:
        """Test that valid JOIN passes verification."""
        sql = """
            SELECT c.name, o.amount
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
        """
        result = syntax_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED

    def test_valid_aggregation(
        self, syntax_verifier: SyntaxVerifier, verification_context: dict
    ) -> None:
        """Test that aggregation queries pass verification."""
        sql = "SELECT COUNT(*) FROM customers"
        result = syntax_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED

    def test_invalid_syntax_typo(
        self, syntax_verifier: SyntaxVerifier, verification_context: dict
    ) -> None:
        """Test that typos are caught."""
        sql = "SELEC name FROM customers"  # Missing 'T'
        result = syntax_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED
        assert "syntax error" in result.message.lower()

    def test_invalid_syntax_missing_from(
        self, syntax_verifier: SyntaxVerifier, verification_context: dict
    ) -> None:
        """Test that missing FROM is caught."""
        sql = "SELECT name customers"
        result = syntax_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED

    def test_invalid_syntax_unclosed_string(
        self, syntax_verifier: SyntaxVerifier, verification_context: dict
    ) -> None:
        """Test that unclosed strings are caught."""
        sql = "SELECT * FROM customers WHERE name = 'test"
        result = syntax_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED


class TestSchemaVerifier:
    """Tests for the SchemaVerifier."""

    def test_valid_table(
        self, schema_verifier: SchemaVerifier, verification_context: dict
    ) -> None:
        """Test that valid tables pass verification."""
        sql = "SELECT * FROM customers"
        result = schema_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED

    def test_valid_columns(
        self, schema_verifier: SchemaVerifier, verification_context: dict
    ) -> None:
        """Test that valid columns pass verification."""
        sql = "SELECT name, email FROM customers"
        result = schema_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED

    def test_invalid_table(
        self, schema_verifier: SchemaVerifier, verification_context: dict
    ) -> None:
        """Test that unknown tables are caught."""
        sql = "SELECT * FROM nonexistent_table"
        result = schema_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED
        assert "Unknown table" in result.message

    def test_invalid_column(
        self, schema_verifier: SchemaVerifier, verification_context: dict
    ) -> None:
        """Test that unknown columns are caught."""
        sql = "SELECT nonexistent_column FROM customers"
        result = schema_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED
        assert "unknown column" in result.message.lower()

    def test_aliased_column_passes(
        self, schema_verifier: SchemaVerifier, verification_context: dict
    ) -> None:
        """Test that aliased columns are not flagged as unknown."""
        sql = "SELECT COUNT(*) as total FROM customers"
        result = schema_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED

    def test_multiple_valid_tables(
        self, schema_verifier: SchemaVerifier, verification_context: dict
    ) -> None:
        """Test JOINs with multiple valid tables."""
        sql = """
            SELECT c.name, o.amount
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
        """
        result = schema_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED


class TestSafetyVerifier:
    """Tests for the SafetyVerifier."""

    def test_safe_select(
        self, safety_verifier: SafetyVerifier, verification_context: dict
    ) -> None:
        """Test that SELECT passes safety check."""
        sql = "SELECT * FROM customers"
        result = safety_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED

    def test_dangerous_drop(
        self, safety_verifier: SafetyVerifier, verification_context: dict
    ) -> None:
        """Test that DROP is caught."""
        sql = "DROP TABLE customers"
        result = safety_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED
        assert "DROP" in result.message

    def test_dangerous_truncate(
        self, safety_verifier: SafetyVerifier, verification_context: dict
    ) -> None:
        """Test that TRUNCATE is caught."""
        sql = "TRUNCATE TABLE customers"
        result = safety_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED
        assert "TRUNCATE" in result.message

    def test_dangerous_delete_no_where(
        self, safety_verifier: SafetyVerifier, verification_context: dict
    ) -> None:
        """Test that DELETE without WHERE is caught."""
        sql = "DELETE FROM customers"
        result = safety_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED
        assert "DELETE" in result.message

    def test_safe_delete_with_where(
        self, safety_verifier: SafetyVerifier, verification_context: dict
    ) -> None:
        """Test that DELETE with WHERE passes."""
        sql = "DELETE FROM customers WHERE id = 1"
        result = safety_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.PASSED

    def test_dangerous_sql_injection(
        self, safety_verifier: SafetyVerifier, verification_context: dict
    ) -> None:
        """Test that SQL injection patterns are caught."""
        sql = "SELECT * FROM customers; -- DROP TABLE users"
        result = safety_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED

    def test_dangerous_drop_database(
        self, safety_verifier: SafetyVerifier, verification_context: dict
    ) -> None:
        """Test that DROP DATABASE is caught."""
        sql = "DROP DATABASE production"
        result = safety_verifier.verify(sql, verification_context)
        assert result.status == VerificationStatus.FAILED


class TestSemanticVerifier:
    """Tests for the SemanticVerifier."""

    def test_aggregation_intent_with_sql(
        self, semantic_verifier: SemanticVerifier, sample_schema: dict
    ) -> None:
        """Test that aggregation intent with matching SQL passes."""
        context = {
            "original_query": "What is the total revenue?",
            "schema": sample_schema,
        }
        sql = "SELECT SUM(amount) FROM orders"
        result = semantic_verifier.verify(sql, context)
        assert result.status == VerificationStatus.PASSED

    def test_aggregation_intent_without_sql(
        self, semantic_verifier: SemanticVerifier, sample_schema: dict
    ) -> None:
        """Test that aggregation intent without aggregation SQL fails."""
        context = {
            "original_query": "What is the total revenue?",
            "schema": sample_schema,
        }
        sql = "SELECT amount FROM orders"
        result = semantic_verifier.verify(sql, context)
        assert result.status == VerificationStatus.FAILED
        assert "aggregation" in result.message.lower()

    def test_ordering_intent_with_sql(
        self, semantic_verifier: SemanticVerifier, sample_schema: dict
    ) -> None:
        """Test that ordering intent with matching SQL passes."""
        context = {
            "original_query": "Show the highest spending customers",
            "schema": sample_schema,
        }
        sql = "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id ORDER BY SUM(amount) DESC"
        result = semantic_verifier.verify(sql, context)
        assert result.status == VerificationStatus.PASSED

    def test_ordering_intent_without_sql(
        self, semantic_verifier: SemanticVerifier, sample_schema: dict
    ) -> None:
        """Test that ordering intent without ORDER BY fails."""
        context = {
            "original_query": "Show the highest spending customers",
            "schema": sample_schema,
        }
        sql = "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id"
        result = semantic_verifier.verify(sql, context)
        assert result.status == VerificationStatus.FAILED
        assert "ORDER BY" in result.message

    def test_limit_intent_with_sql(
        self, semantic_verifier: SemanticVerifier, sample_schema: dict
    ) -> None:
        """Test that limit intent with LIMIT passes."""
        context = {
            "original_query": "Show top 5 customers",
            "schema": sample_schema,
        }
        sql = "SELECT name FROM customers ORDER BY id LIMIT 5"
        result = semantic_verifier.verify(sql, context)
        assert result.status == VerificationStatus.PASSED

    def test_limit_intent_without_sql(
        self, semantic_verifier: SemanticVerifier, sample_schema: dict
    ) -> None:
        """Test that limit intent without LIMIT fails."""
        context = {
            "original_query": "Show top 5 customers",
            "schema": sample_schema,
        }
        sql = "SELECT name FROM customers ORDER BY id"
        result = semantic_verifier.verify(sql, context)
        assert result.status == VerificationStatus.FAILED
        assert "LIMIT" in result.message

    def test_no_special_intent(
        self, semantic_verifier: SemanticVerifier, sample_schema: dict
    ) -> None:
        """Test that queries without special intent pass."""
        context = {
            "original_query": "Show all customers",
            "schema": sample_schema,
        }
        sql = "SELECT * FROM customers"
        result = semantic_verifier.verify(sql, context)
        assert result.status == VerificationStatus.PASSED


class TestVerificationChain:
    """Tests for the full verification chain."""

    def test_all_pass(
        self, verification_chain, verification_context: dict
    ) -> None:
        """Test that valid SQL passes all verifiers."""
        sql = "SELECT name, email FROM customers WHERE tier = 'premium'"
        passed, results = verification_chain.run(sql, verification_context)
        assert passed is True
        assert len(results) == 4  # All verifiers run
        assert all(r.status == VerificationStatus.PASSED for r in results)

    def test_fail_fast_on_syntax(
        self, verification_chain, verification_context: dict
    ) -> None:
        """Test that chain fails fast on syntax error."""
        sql = "SELEC * FROM customers"  # Typo
        passed, results = verification_chain.run(sql, verification_context)
        assert passed is False
        assert len(results) == 1  # Stopped at first failure
        assert results[0].verifier_name == "SyntaxVerifier"

    def test_fail_on_safety(
        self, verification_chain, verification_context: dict
    ) -> None:
        """Test that chain fails on safety violation."""
        sql = "DELETE FROM customers"
        passed, results = verification_chain.run(sql, verification_context)
        assert passed is False
        # Should fail at safety (after syntax and schema pass)
        failed_verifier = next(r for r in results if r.status == VerificationStatus.FAILED)
        assert failed_verifier.verifier_name == "SafetyVerifier"
