"""
Schema Verifier
===============

Validates that referenced tables and columns exist in the schema.
"""

import re

from nl_to_sql.models import VerificationResult, VerificationStatus
from nl_to_sql.verifiers.base import Verifier
from nl_to_sql.verifiers.syntax import SAMPLE_SCHEMA


class SchemaVerifier(Verifier):
    """Validates that referenced tables and columns exist in schema."""

    @property
    def name(self) -> str:
        return "SchemaVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        """
        Verify all referenced tables and columns exist in the schema.

        Args:
            sql: SQL query to validate
            context: Must contain 'schema' key with table definitions

        Returns:
            VerificationResult with PASSED or FAILED status
        """
        schema = context.get("schema", SAMPLE_SCHEMA)
        sql_lower = sql.lower()

        errors = []

        # Extract table references (simplified parser)
        table_pattern = r"\b(?:FROM|JOIN|INTO|UPDATE)\s+(\w+)"
        referenced_tables = re.findall(table_pattern, sql_lower, re.IGNORECASE)

        for table in referenced_tables:
            if table not in schema:
                errors.append(f"Unknown table: '{table}'")

        # Extract column references (simplified - checks against all known columns)
        all_columns = set()
        for table_info in schema.values():
            all_columns.update(table_info["columns"])

        # Extract aliases defined with AS keyword
        alias_pattern = r"\bAS\s+(\w+)"
        defined_aliases = set(re.findall(alias_pattern, sql_lower, re.IGNORECASE))

        # SQL keywords and functions to ignore
        sql_keywords = {
            "*",
            "count",
            "sum",
            "avg",
            "max",
            "min",
            "as",
            "distinct",
            "from",
            "where",
            "and",
            "or",
            "not",
            "in",
            "like",
            "between",
            "group",
            "by",
            "order",
            "having",
            "asc",
            "desc",
            "limit",
            "join",
            "left",
            "right",
            "inner",
            "outer",
            "on",
            "null",
        }

        # Find potential column references (word.word or standalone words after SELECT)
        select_match = re.search(
            r"SELECT\s+(.+?)\s+FROM", sql_lower, re.IGNORECASE | re.DOTALL
        )
        if select_match:
            select_clause = select_match.group(1)
            # Handle table.column notation - extract just the column part
            col_refs = re.findall(r"(?:\w+\.)?(\w+)", select_clause)
            for col in col_refs:
                col_lower = col.lower()
                # Skip if: known column, defined alias, SQL keyword, or number
                if (
                    col_lower not in all_columns
                    and col_lower not in defined_aliases
                    and col_lower not in sql_keywords
                    and not re.match(r"^\d+$", col)
                ):
                    errors.append(f"Potentially unknown column: '{col}'")

        if errors:
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message=f"Schema validation failed: {'; '.join(errors)}",
                details={"errors": errors, "available_tables": list(schema.keys())},
            )

        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.PASSED,
            message="All referenced tables and columns are valid",
        )
