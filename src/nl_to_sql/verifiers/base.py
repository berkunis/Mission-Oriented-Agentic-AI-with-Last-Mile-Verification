"""
Base Verifier Classes
=====================

Abstract base class and verification chain implementation.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from nl_to_sql.models import VerificationResult, VerificationStatus

if TYPE_CHECKING:
    from nl_to_sql.verifiers.syntax import SyntaxVerifier
    from nl_to_sql.verifiers.schema import SchemaVerifier
    from nl_to_sql.verifiers.safety import SafetyVerifier
    from nl_to_sql.verifiers.semantic import SemanticVerifier


class Verifier(ABC):
    """Base class for all verifiers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this verifier."""
        pass

    @abstractmethod
    def verify(self, sql: str, context: dict) -> VerificationResult:
        """
        Verify the SQL against this verifier's rules.

        Args:
            sql: The SQL query to verify
            context: Additional context (schema, original query, etc.)

        Returns:
            VerificationResult indicating pass/fail with details
        """
        pass


class VerificationChain:
    """Runs all verifiers in sequence, collecting results."""

    def __init__(self, verifiers: list[Verifier] | None = None) -> None:
        """
        Initialize the verification chain.

        Args:
            verifiers: List of verifiers to run. Defaults to standard chain.
        """
        if verifiers is not None:
            self.verifiers = verifiers
        else:
            # Lazy import to avoid circular imports
            from nl_to_sql.verifiers.syntax import SyntaxVerifier
            from nl_to_sql.verifiers.schema import SchemaVerifier
            from nl_to_sql.verifiers.safety import SafetyVerifier
            from nl_to_sql.verifiers.semantic import SemanticVerifier

            self.verifiers = [
                SyntaxVerifier(),
                SchemaVerifier(),
                SafetyVerifier(),
                SemanticVerifier(),
            ]

    def run(self, sql: str, context: dict) -> tuple[bool, list[VerificationResult]]:
        """
        Run all verifiers. Returns (all_passed, results).

        Stops at first failure for efficiency.

        Args:
            sql: The SQL query to verify
            context: Additional context for verification

        Returns:
            Tuple of (success, list of verification results)
        """
        results = []

        for verifier in self.verifiers:
            result = verifier.verify(sql, context)
            results.append(result)

            if result.status == VerificationStatus.FAILED:
                return False, results

        return True, results
