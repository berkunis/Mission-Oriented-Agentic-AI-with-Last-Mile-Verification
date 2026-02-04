"""
Safety Verifier
===============

Ensures no destructive or dangerous operations are present.
"""

import re

from nl_to_sql.models import VerificationResult, VerificationStatus
from nl_to_sql.verifiers.base import Verifier


class SafetyVerifier(Verifier):
    """Ensures no destructive operations are present."""

    DANGEROUS_PATTERNS = [
        (r"\bDROP\s+(?:TABLE|DATABASE|INDEX)", "DROP operation detected"),
        (r"\bTRUNCATE\s+", "TRUNCATE operation detected"),
        (r"\bDELETE\s+FROM\s+\w+\s*(?:;|$)", "DELETE without WHERE clause"),
        (
            r"\bUPDATE\s+\w+\s+SET\s+.+(?:;|$)(?!.*WHERE)",
            "UPDATE without WHERE clause",
        ),
        (r";\s*--", "SQL comment after statement (potential injection)"),
        (r"\bEXEC\s*\(", "Dynamic SQL execution detected"),
    ]

    @property
    def name(self) -> str:
        return "SafetyVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        """
        Verify SQL does not contain dangerous operations.

        Args:
            sql: SQL query to validate
            context: Additional context (unused for this verifier)

        Returns:
            VerificationResult with PASSED or FAILED status
        """
        violations = []

        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                violations.append(description)

        if violations:
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message=f"Safety check failed: {'; '.join(violations)}",
                details={"violations": violations},
            )

        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.PASSED,
            message="No dangerous operations detected",
        )
