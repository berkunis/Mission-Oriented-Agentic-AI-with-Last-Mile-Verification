"""
PII Detection
=============

Personally Identifiable Information detection for SQL queries.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Pattern

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nl_to_sql.models import VerificationResult, VerificationStatus
from nl_to_sql.verifiers.base import Verifier


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    NAME = "name"
    ADDRESS = "address"


@dataclass
class PIIMatch:
    """A detected PII match."""

    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float


class PIIDetector:
    """
    Detects PII in text using pattern matching.

    This is a simplified implementation for demonstration.
    Production systems should use more sophisticated NLP-based detection.
    """

    # PII detection patterns
    PATTERNS: dict[PIIType, tuple[Pattern, float]] = {
        PIIType.EMAIL: (
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            0.95,
        ),
        PIIType.PHONE: (
            re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
            0.85,
        ),
        PIIType.SSN: (
            re.compile(r"\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b"),
            0.90,
        ),
        PIIType.CREDIT_CARD: (
            re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"),
            0.95,
        ),
        PIIType.IP_ADDRESS: (
            re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
            0.80,
        ),
    }

    # Sensitive column patterns
    SENSITIVE_COLUMNS = {
        "ssn",
        "social_security",
        "social_security_number",
        "tax_id",
        "tin",
        "credit_card",
        "card_number",
        "cvv",
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "private_key",
        "date_of_birth",
        "dob",
        "birthdate",
        "salary",
        "income",
        "bank_account",
        "routing_number",
    }

    def __init__(self, min_confidence: float = 0.8):
        """
        Initialize PII detector.

        Args:
            min_confidence: Minimum confidence threshold for matches
        """
        self.min_confidence = min_confidence

    def detect(self, text: str) -> list[PIIMatch]:
        """
        Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            List of PII matches found
        """
        matches = []

        for pii_type, (pattern, confidence) in self.PATTERNS.items():
            if confidence < self.min_confidence:
                continue

            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                    )
                )

        return matches

    def detect_sensitive_columns(self, sql: str) -> list[str]:
        """
        Detect references to sensitive columns in SQL.

        Args:
            sql: SQL query to check

        Returns:
            List of sensitive column names found
        """
        sql_lower = sql.lower()
        found = []

        for col in self.SENSITIVE_COLUMNS:
            # Check for column reference patterns
            patterns = [
                rf"\b{col}\b",  # Exact match
                rf"\.{col}\b",  # table.column
                rf"\[{col}\]",  # [column] notation
                rf'"{col}"',  # "column" notation
            ]
            for pattern in patterns:
                if re.search(pattern, sql_lower):
                    found.append(col)
                    break

        return found

    def redact(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Text containing PII

        Returns:
            Text with PII redacted
        """
        matches = self.detect(text)

        # Sort by position (descending) to replace from end
        matches.sort(key=lambda m: m.start, reverse=True)

        result = text
        for match in matches:
            redacted = f"[{match.pii_type.value.upper()}_REDACTED]"
            result = result[:match.start] + redacted + result[match.end:]

        return result


class PIIVerifier(Verifier):
    """
    Verifier that checks for PII in SQL queries.

    Can be configured to:
    - Warn about PII in query values
    - Block access to sensitive columns
    - Redact PII from audit logs
    """

    def __init__(
        self,
        block_pii_values: bool = True,
        block_sensitive_columns: bool = True,
        allowed_columns: set[str] | None = None,
    ):
        """
        Initialize PII verifier.

        Args:
            block_pii_values: Block queries containing PII values
            block_sensitive_columns: Block access to sensitive columns
            allowed_columns: Whitelist of sensitive columns that are allowed
        """
        self.detector = PIIDetector()
        self.block_pii_values = block_pii_values
        self.block_sensitive_columns = block_sensitive_columns
        self.allowed_columns = allowed_columns or set()

    @property
    def name(self) -> str:
        return "PIIVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        """
        Verify SQL does not expose PII.

        Args:
            sql: SQL query to verify
            context: Verification context

        Returns:
            VerificationResult indicating pass/fail
        """
        issues = []
        details = {}

        # Check for PII values in the query
        if self.block_pii_values:
            pii_matches = self.detector.detect(sql)
            if pii_matches:
                pii_types = list(set(m.pii_type.value for m in pii_matches))
                issues.append(f"Query contains PII values: {', '.join(pii_types)}")
                details["pii_types_found"] = pii_types

        # Check for sensitive column access
        if self.block_sensitive_columns:
            sensitive_cols = self.detector.detect_sensitive_columns(sql)
            blocked_cols = [c for c in sensitive_cols if c not in self.allowed_columns]
            if blocked_cols:
                issues.append(f"Query accesses sensitive columns: {', '.join(blocked_cols)}")
                details["sensitive_columns"] = blocked_cols

        if issues:
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message=f"PII/sensitive data check failed: {'; '.join(issues)}",
                details=details,
            )

        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.PASSED,
            message="No PII or sensitive data issues detected",
        )
