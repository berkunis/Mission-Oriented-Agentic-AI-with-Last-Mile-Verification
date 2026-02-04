"""
Audit Export
============

NIST-compliant audit trail export for compliance and forensics.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nl_to_sql.models import AgentResult, AuditEntry, VerificationStatus


class AuditEventType(str, Enum):
    """Types of audit events."""

    QUERY_SUBMITTED = "query_submitted"
    SQL_GENERATED = "sql_generated"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    CORRECTION_ATTEMPTED = "correction_attempted"
    QUERY_COMPLETED = "query_completed"
    QUERY_FAILED = "query_failed"
    ACCESS_DENIED = "access_denied"
    RATE_LIMITED = "rate_limited"


@dataclass
class NISTAuditRecord:
    """
    NIST SP 800-53 compliant audit record.

    Based on AU-3 (Content of Audit Records) requirements:
    - Type of event
    - When the event occurred
    - Where the event occurred
    - Source of the event
    - Outcome of the event
    - Identity of individuals/subjects associated with the event
    """

    # Required fields (AU-3)
    event_id: str
    event_type: str
    timestamp: str
    source_component: str
    outcome: str
    subject_id: str

    # Additional context
    event_details: dict[str, Any]
    request_id: str | None = None
    session_id: str | None = None

    # Integrity
    record_hash: str | None = None
    previous_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NISTAuditRecord":
        """Create from dictionary."""
        return cls(**data)


class NISTAuditFormatter:
    """
    Formats audit records according to NIST guidelines.

    Implements:
    - AU-3: Content of Audit Records
    - AU-8: Time Stamps
    - AU-9: Protection of Audit Information (via hashing)
    - AU-10: Non-repudiation (via chain of hashes)
    """

    def __init__(self, component_name: str = "nl-to-sql-agent"):
        """
        Initialize formatter.

        Args:
            component_name: Name of the component generating audit records
        """
        self.component_name = component_name
        self._last_hash: str | None = None

    def format_agent_result(
        self,
        result: AgentResult,
        subject_id: str,
        request_id: str | None = None,
    ) -> list[NISTAuditRecord]:
        """
        Convert an AgentResult to NIST-compliant audit records.

        Args:
            result: Agent result to convert
            subject_id: Identifier of the user/system that initiated the request
            request_id: Optional request correlation ID

        Returns:
            List of NIST audit records
        """
        records = []

        # Query submitted event
        records.append(
            self._create_record(
                event_type=AuditEventType.QUERY_SUBMITTED,
                outcome="initiated",
                subject_id=subject_id,
                request_id=request_id,
                details={
                    "original_query": result.original_query,
                },
            )
        )

        # Process audit trail entries
        for entry in result.audit_trail:
            records.extend(
                self._format_audit_entry(entry, subject_id, request_id)
            )

        # Final outcome event
        final_event = (
            AuditEventType.QUERY_COMPLETED
            if result.success
            else AuditEventType.QUERY_FAILED
        )
        records.append(
            self._create_record(
                event_type=final_event,
                outcome="success" if result.success else "failure",
                subject_id=subject_id,
                request_id=request_id,
                details={
                    "success": result.success,
                    "attempts": result.attempts,
                    "final_sql": result.sql,
                    "message": result.final_message,
                },
            )
        )

        return records

    def _format_audit_entry(
        self,
        entry: AuditEntry,
        subject_id: str,
        request_id: str | None,
    ) -> list[NISTAuditRecord]:
        """Format a single audit entry."""
        records = []

        # Determine event type based on step
        if "generation" in entry.step:
            event_type = AuditEventType.SQL_GENERATED
        elif "verification" in entry.step:
            # Check verification results
            if entry.verification_results:
                has_failure = any(
                    vr.status == VerificationStatus.FAILED
                    for vr in entry.verification_results
                )
                event_type = (
                    AuditEventType.VERIFICATION_FAILED
                    if has_failure
                    else AuditEventType.VERIFICATION_PASSED
                )
            else:
                event_type = AuditEventType.VERIFICATION_STARTED
        elif "llm_response" in entry.step:
            event_type = AuditEventType.SQL_GENERATED
        else:
            event_type = AuditEventType.QUERY_SUBMITTED

        # Build details
        details = {
            "step": entry.step,
            **entry.input_data,
            **entry.output_data,
        }

        # Add verification results if present
        if entry.verification_results:
            details["verification_results"] = [
                {
                    "verifier": vr.verifier_name,
                    "status": vr.status.value,
                    "message": vr.message,
                }
                for vr in entry.verification_results
            ]

        # Determine outcome
        if event_type == AuditEventType.VERIFICATION_FAILED:
            outcome = "failure"
        elif event_type == AuditEventType.VERIFICATION_PASSED:
            outcome = "success"
        else:
            outcome = "completed"

        records.append(
            self._create_record(
                event_type=event_type,
                outcome=outcome,
                subject_id=subject_id,
                request_id=request_id,
                details=details,
                timestamp=entry.timestamp,
            )
        )

        return records

    def _create_record(
        self,
        event_type: AuditEventType,
        outcome: str,
        subject_id: str,
        request_id: str | None,
        details: dict[str, Any],
        timestamp: str | None = None,
    ) -> NISTAuditRecord:
        """Create a NIST audit record with integrity hash."""
        event_id = str(uuid4())

        record = NISTAuditRecord(
            event_id=event_id,
            event_type=event_type.value,
            timestamp=timestamp or datetime.utcnow().isoformat(),
            source_component=self.component_name,
            outcome=outcome,
            subject_id=subject_id,
            event_details=details,
            request_id=request_id,
            previous_hash=self._last_hash,
        )

        # Calculate record hash for integrity
        hash_content = json.dumps(record.to_dict(), sort_keys=True)
        record.record_hash = hashlib.sha256(hash_content.encode()).hexdigest()

        # Update chain
        self._last_hash = record.record_hash

        return record


class AuditExporter:
    """
    Exports audit records to various formats.

    Supports:
    - JSON (default)
    - JSON Lines (for streaming)
    - CSV (for spreadsheet analysis)
    """

    def __init__(self, formatter: NISTAuditFormatter | None = None):
        """
        Initialize exporter.

        Args:
            formatter: Optional custom formatter
        """
        self.formatter = formatter or NISTAuditFormatter()

    def export_to_json(
        self,
        records: list[NISTAuditRecord],
        output: TextIO | Path | str,
    ) -> None:
        """
        Export records to JSON file.

        Args:
            records: Audit records to export
            output: File path or file object
        """
        data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "record_count": len(records),
            "records": [r.to_dict() for r in records],
        }

        if isinstance(output, (str, Path)):
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
        else:
            json.dump(data, output, indent=2)

    def export_to_jsonl(
        self,
        records: list[NISTAuditRecord],
        output: TextIO | Path | str,
    ) -> None:
        """
        Export records to JSON Lines format.

        One record per line for streaming processing.
        """
        if isinstance(output, (str, Path)):
            f = open(output, "w")
            should_close = True
        else:
            f = output
            should_close = False

        try:
            for record in records:
                f.write(json.dumps(record.to_dict()) + "\n")
        finally:
            if should_close:
                f.close()

    def export_agent_result(
        self,
        result: AgentResult,
        subject_id: str,
        output: TextIO | Path | str,
        request_id: str | None = None,
        format: str = "json",
    ) -> list[NISTAuditRecord]:
        """
        Export an agent result's audit trail.

        Args:
            result: Agent result to export
            subject_id: User/system identifier
            output: Output file or path
            request_id: Optional request correlation ID
            format: Output format ("json" or "jsonl")

        Returns:
            List of exported audit records
        """
        records = self.formatter.format_agent_result(result, subject_id, request_id)

        if format == "jsonl":
            self.export_to_jsonl(records, output)
        else:
            self.export_to_json(records, output)

        return records


def verify_audit_chain(records: list[NISTAuditRecord]) -> tuple[bool, list[str]]:
    """
    Verify the integrity of an audit record chain.

    Args:
        records: List of audit records to verify

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []
    previous_hash = None

    for i, record in enumerate(records):
        # Verify previous hash chain
        if record.previous_hash != previous_hash:
            errors.append(
                f"Record {i} ({record.event_id}): Chain broken - "
                f"expected previous_hash {previous_hash}, got {record.previous_hash}"
            )

        # Verify record hash
        temp_record = NISTAuditRecord(**record.to_dict())
        temp_record.record_hash = None
        hash_content = json.dumps(temp_record.to_dict(), sort_keys=True)
        expected_hash = hashlib.sha256(hash_content.encode()).hexdigest()

        if record.record_hash != expected_hash:
            errors.append(
                f"Record {i} ({record.event_id}): Hash mismatch - "
                f"record may have been tampered with"
            )

        previous_hash = record.record_hash

    return len(errors) == 0, errors
