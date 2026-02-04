"""
Syntax Verifier
===============

Validates SQL syntax using SQLite's parser.
"""

import sqlite3

from nl_to_sql.models import VerificationResult, VerificationStatus
from nl_to_sql.verifiers.base import Verifier


# Default schema for validation
SAMPLE_SCHEMA = {
    "customers": {
        "columns": ["id", "name", "email", "created_at", "tier"],
        "types": {
            "id": "INTEGER",
            "name": "TEXT",
            "email": "TEXT",
            "created_at": "DATE",
            "tier": "TEXT",
        },
    },
    "orders": {
        "columns": ["id", "customer_id", "amount", "order_date", "status"],
        "types": {
            "id": "INTEGER",
            "customer_id": "INTEGER",
            "amount": "DECIMAL",
            "order_date": "DATE",
            "status": "TEXT",
        },
    },
    "products": {
        "columns": ["id", "name", "price", "category", "stock"],
        "types": {
            "id": "INTEGER",
            "name": "TEXT",
            "price": "DECIMAL",
            "category": "TEXT",
            "stock": "INTEGER",
        },
    },
}


class SyntaxVerifier(Verifier):
    """Validates SQL syntax using SQLite's parser with schema tables."""

    @property
    def name(self) -> str:
        return "SyntaxVerifier"

    def _create_schema_tables(
        self, conn: sqlite3.Connection, schema: dict
    ) -> None:
        """Create empty tables matching the schema for syntax validation."""
        cursor = conn.cursor()
        for table_name, table_info in schema.items():
            columns = []
            for col_name in table_info["columns"]:
                col_type = table_info["types"].get(col_name, "TEXT")
                columns.append(f"{col_name} {col_type}")
            create_sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"
            cursor.execute(create_sql)
        conn.commit()

    def verify(self, sql: str, context: dict) -> VerificationResult:
        """
        Verify SQL syntax is valid.

        Args:
            sql: SQL query to validate
            context: Must contain 'schema' key with table definitions

        Returns:
            VerificationResult with PASSED or FAILED status
        """
        try:
            schema = context.get("schema", SAMPLE_SCHEMA)
            # Create in-memory database with schema tables
            conn = sqlite3.connect(":memory:")
            self._create_schema_tables(conn, schema)

            # Use EXPLAIN to validate syntax
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.PASSED,
                message="SQL syntax is valid",
            )
        except sqlite3.Error as e:
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message=f"SQL syntax error: {e!s}",
                details={"error_type": type(e).__name__, "error_message": str(e)},
            )
