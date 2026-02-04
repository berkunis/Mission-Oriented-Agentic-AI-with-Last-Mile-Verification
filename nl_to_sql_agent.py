"""
Mission-Oriented Agentic AI with Last-Mile Verification
========================================================
A prototype demonstrating that agentic AI systems require verification
and correction mechanisms to be viable in mission-critical environments.

Use case: Natural Language to SQL query generation with verifiable outputs.
"""

import json
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional


# =============================================================================
# Data Structures
# =============================================================================

class VerificationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VerificationResult:
    """Result of a single verification step."""
    verifier_name: str
    status: VerificationStatus
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AuditEntry:
    """Single entry in the audit trail."""
    timestamp: str
    step: str
    input_data: dict
    output_data: dict
    verification_results: list[VerificationResult] = field(default_factory=list)


@dataclass
class AgentResult:
    """Final result from the agent."""
    success: bool
    sql: Optional[str]
    original_query: str
    attempts: int
    audit_trail: list[AuditEntry]
    final_message: str


# =============================================================================
# Schema Definition (for validation)
# =============================================================================

SAMPLE_SCHEMA = {
    "customers": {
        "columns": ["id", "name", "email", "created_at", "tier"],
        "types": {"id": "INTEGER", "name": "TEXT", "email": "TEXT",
                  "created_at": "DATE", "tier": "TEXT"}
    },
    "orders": {
        "columns": ["id", "customer_id", "amount", "order_date", "status"],
        "types": {"id": "INTEGER", "customer_id": "INTEGER", "amount": "DECIMAL",
                  "order_date": "DATE", "status": "TEXT"}
    },
    "products": {
        "columns": ["id", "name", "price", "category", "stock"],
        "types": {"id": "INTEGER", "name": "TEXT", "price": "DECIMAL",
                  "category": "TEXT", "stock": "INTEGER"}
    }
}


# =============================================================================
# Verifiers (Last-Mile Verification Chain)
# =============================================================================

class Verifier(ABC):
    """Base class for all verifiers."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def verify(self, sql: str, context: dict) -> VerificationResult:
        pass


class SyntaxVerifier(Verifier):
    """Validates SQL syntax using SQLite's parser with schema tables."""

    @property
    def name(self) -> str:
        return "SyntaxVerifier"

    def _create_schema_tables(self, conn: sqlite3.Connection, schema: dict):
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
                message="SQL syntax is valid"
            )
        except sqlite3.Error as e:
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message=f"SQL syntax error: {str(e)}",
                details={"error_type": type(e).__name__, "error_message": str(e)}
            )


class SchemaVerifier(Verifier):
    """Validates that referenced tables and columns exist in schema."""

    @property
    def name(self) -> str:
        return "SchemaVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        schema = context.get("schema", SAMPLE_SCHEMA)
        sql_upper = sql.upper()
        sql_lower = sql.lower()

        errors = []

        # Extract table references (simplified parser)
        table_pattern = r'\b(?:FROM|JOIN|INTO|UPDATE)\s+(\w+)'
        referenced_tables = re.findall(table_pattern, sql_lower, re.IGNORECASE)

        for table in referenced_tables:
            if table not in schema:
                errors.append(f"Unknown table: '{table}'")

        # Extract column references (simplified - checks against all known columns)
        all_columns = set()
        for table_info in schema.values():
            all_columns.update(table_info["columns"])

        # Extract aliases defined with AS keyword
        alias_pattern = r'\bAS\s+(\w+)'
        defined_aliases = set(re.findall(alias_pattern, sql_lower, re.IGNORECASE))

        # SQL keywords and functions to ignore
        sql_keywords = {'*', 'count', 'sum', 'avg', 'max', 'min', 'as', 'distinct',
                       'from', 'where', 'and', 'or', 'not', 'in', 'like', 'between',
                       'group', 'by', 'order', 'having', 'asc', 'desc', 'limit',
                       'join', 'left', 'right', 'inner', 'outer', 'on', 'null'}

        # Find potential column references (word.word or standalone words after SELECT)
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql_lower, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Handle table.column notation - extract just the column part
            col_refs = re.findall(r'(?:\w+\.)?(\w+)', select_clause)
            for col in col_refs:
                col_lower = col.lower()
                # Skip if: known column, defined alias, SQL keyword, or number
                if (col_lower not in all_columns and
                    col_lower not in defined_aliases and
                    col_lower not in sql_keywords and
                    not re.match(r'^\d+$', col)):
                    errors.append(f"Potentially unknown column: '{col}'")

        if errors:
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message=f"Schema validation failed: {'; '.join(errors)}",
                details={"errors": errors, "available_tables": list(schema.keys())}
            )

        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.PASSED,
            message="All referenced tables and columns are valid"
        )


class SafetyVerifier(Verifier):
    """Ensures no destructive operations are present."""

    DANGEROUS_PATTERNS = [
        (r'\bDROP\s+(?:TABLE|DATABASE|INDEX)', "DROP operation detected"),
        (r'\bTRUNCATE\s+', "TRUNCATE operation detected"),
        (r'\bDELETE\s+FROM\s+\w+\s*(?:;|$)', "DELETE without WHERE clause"),
        (r'\bUPDATE\s+\w+\s+SET\s+.+(?:;|$)(?!.*WHERE)', "UPDATE without WHERE clause"),
        (r';\s*--', "SQL comment after statement (potential injection)"),
        (r'\bEXEC\s*\(', "Dynamic SQL execution detected"),
    ]

    @property
    def name(self) -> str:
        return "SafetyVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        violations = []

        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                violations.append(description)

        if violations:
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message=f"Safety check failed: {'; '.join(violations)}",
                details={"violations": violations}
            )

        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.PASSED,
            message="No dangerous operations detected"
        )


class SemanticVerifier(Verifier):
    """
    Validates that the SQL appears to match the user's intent.
    This is a simplified heuristic check - in production, this could
    use another LLM call for semantic validation.
    """

    @property
    def name(self) -> str:
        return "SemanticVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        original_query = context.get("original_query", "").lower()
        sql_lower = sql.lower()

        issues = []

        # Check for aggregation keywords
        agg_keywords = ["total", "sum", "count", "average", "avg", "how many"]
        has_agg_intent = any(kw in original_query for kw in agg_keywords)
        has_agg_sql = any(fn in sql_lower for fn in ["count(", "sum(", "avg(", "max(", "min("])

        if has_agg_intent and not has_agg_sql:
            issues.append("Query seems to request aggregation but SQL has none")

        # Check for ordering keywords
        order_keywords = ["highest", "lowest", "top", "bottom", "most", "least", "largest", "smallest"]
        has_order_intent = any(kw in original_query for kw in order_keywords)
        has_order_sql = "order by" in sql_lower

        if has_order_intent and not has_order_sql:
            issues.append("Query seems to request ordering but SQL has no ORDER BY")

        # Check for limit keywords
        limit_keywords = ["top", "first", "last", "only"]
        has_limit_intent = any(kw in original_query for kw in limit_keywords)
        has_limit_sql = "limit" in sql_lower

        if has_limit_intent and not has_limit_sql:
            issues.append("Query seems to request limited results but SQL has no LIMIT")

        if issues:
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message=f"Semantic mismatch: {'; '.join(issues)}",
                details={"issues": issues}
            )

        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.PASSED,
            message="SQL appears to match query intent"
        )


# =============================================================================
# Verification Chain
# =============================================================================

class VerificationChain:
    """Runs all verifiers in sequence, collecting results."""

    def __init__(self, verifiers: list[Verifier] = None):
        self.verifiers = verifiers or [
            SyntaxVerifier(),
            SchemaVerifier(),
            SafetyVerifier(),
            SemanticVerifier(),
        ]

    def run(self, sql: str, context: dict) -> tuple[bool, list[VerificationResult]]:
        """
        Run all verifiers. Returns (all_passed, results).
        Stops at first failure for efficiency.
        """
        results = []

        for verifier in self.verifiers:
            result = verifier.verify(sql, context)
            results.append(result)

            if result.status == VerificationStatus.FAILED:
                return False, results

        return True, results


# =============================================================================
# LLM Interface (Pluggable)
# =============================================================================

@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    tokens_used: int = 0


class LLMInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        pass


class MockLLM(LLMInterface):
    """
    Mock LLM for demonstration purposes.
    In production, replace with OpenAI, Anthropic, or other provider.
    """

    def __init__(self, responses: dict[str, list[str]] = None):
        """
        Initialize with canned responses.
        responses: dict mapping query substrings to list of SQL attempts
        """
        self.responses = responses or {}
        self.call_counts = {}

    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        # Find matching response based on prompt content
        for key, sql_attempts in self.responses.items():
            if key.lower() in prompt.lower():
                count = self.call_counts.get(key, 0)
                self.call_counts[key] = count + 1

                # Return successive attempts (simulating correction)
                attempt_idx = min(count, len(sql_attempts) - 1)
                return LLMResponse(
                    content=sql_attempts[attempt_idx],
                    model="mock-llm-v1"
                )

        # Default fallback
        return LLMResponse(
            content="SELECT * FROM unknown_table",
            model="mock-llm-v1"
        )


# =============================================================================
# Agent Controller
# =============================================================================

class NLToSQLAgent:
    """
    Main agent that orchestrates NL-to-SQL conversion with verification.
    """

    SYSTEM_PROMPT = """You are a SQL query generator. Convert natural language
questions into SQL queries for the following schema:

Tables:
- customers (id, name, email, created_at, tier)
- orders (id, customer_id, amount, order_date, status)
- products (id, name, price, category, stock)

Rules:
- Generate only SELECT queries unless explicitly asked otherwise
- Use proper JOIN syntax when relating tables
- Include appropriate WHERE clauses for filtering
- Use aggregation functions (COUNT, SUM, AVG) when quantities are requested
- Add ORDER BY and LIMIT when ranking or "top N" is requested

Return ONLY the SQL query, no explanations."""

    CORRECTION_PROMPT_TEMPLATE = """The previous SQL query had an error.

Original question: {original_query}
Previous SQL: {previous_sql}
Error: {error_message}

Please generate a corrected SQL query that fixes this issue.
Return ONLY the SQL query, no explanations."""

    def __init__(
        self,
        llm: LLMInterface,
        verification_chain: VerificationChain = None,
        max_retries: int = 3,
        schema: dict = None
    ):
        self.llm = llm
        self.verification_chain = verification_chain or VerificationChain()
        self.max_retries = max_retries
        self.schema = schema or SAMPLE_SCHEMA
        self.audit_trail: list[AuditEntry] = []

    def _log_audit(
        self,
        step: str,
        input_data: dict,
        output_data: dict,
        verification_results: list[VerificationResult] = None
    ):
        """Add entry to audit trail."""
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            step=step,
            input_data=input_data,
            output_data=output_data,
            verification_results=verification_results or []
        )
        self.audit_trail.append(entry)

    def _extract_sql(self, llm_output: str) -> str:
        """Extract SQL from LLM output, handling markdown code blocks."""
        # Remove markdown code blocks if present
        sql = llm_output.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            # Remove first and last lines (code block markers)
            sql = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return sql.strip()

    def process(self, natural_language_query: str) -> AgentResult:
        """
        Main entry point: convert NL query to verified SQL.
        """
        self.audit_trail = []  # Reset for new query

        context = {
            "original_query": natural_language_query,
            "schema": self.schema
        }

        # Initial generation
        prompt = f"{self.SYSTEM_PROMPT}\n\nQuestion: {natural_language_query}"

        for attempt in range(self.max_retries + 1):
            # Generate SQL
            self._log_audit(
                step=f"generation_attempt_{attempt + 1}",
                input_data={"prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt},
                output_data={}
            )

            response = self.llm.generate(prompt)
            sql = self._extract_sql(response.content)

            self._log_audit(
                step=f"llm_response_{attempt + 1}",
                input_data={},
                output_data={"sql": sql, "model": response.model}
            )

            # Verify
            passed, verification_results = self.verification_chain.run(sql, context)

            self._log_audit(
                step=f"verification_{attempt + 1}",
                input_data={"sql": sql},
                output_data={"passed": passed},
                verification_results=verification_results
            )

            if passed:
                return AgentResult(
                    success=True,
                    sql=sql,
                    original_query=natural_language_query,
                    attempts=attempt + 1,
                    audit_trail=self.audit_trail,
                    final_message=f"Query verified successfully after {attempt + 1} attempt(s)"
                )

            # Prepare correction prompt for next attempt
            failed_result = next(r for r in verification_results
                               if r.status == VerificationStatus.FAILED)

            prompt = self.CORRECTION_PROMPT_TEMPLATE.format(
                original_query=natural_language_query,
                previous_sql=sql,
                error_message=failed_result.message
            )

        # Max retries exceeded
        return AgentResult(
            success=False,
            sql=None,
            original_query=natural_language_query,
            attempts=self.max_retries + 1,
            audit_trail=self.audit_trail,
            final_message=f"Failed to generate valid SQL after {self.max_retries + 1} attempts"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def print_audit_trail(audit_trail: list[AuditEntry], verbose: bool = False):
    """Pretty-print the audit trail for inspection."""
    print("\n" + "="*60)
    print("AUDIT TRAIL")
    print("="*60)

    for i, entry in enumerate(audit_trail):
        print(f"\n[{i+1}] {entry.step} @ {entry.timestamp}")

        if entry.output_data.get("sql"):
            print(f"    SQL: {entry.output_data['sql']}")

        if entry.verification_results:
            for vr in entry.verification_results:
                status_icon = "✓" if vr.status == VerificationStatus.PASSED else "✗"
                print(f"    {status_icon} {vr.verifier_name}: {vr.message}")

        if verbose and entry.input_data:
            print(f"    Input: {json.dumps(entry.input_data, indent=2)[:200]}")


def print_result(result: AgentResult):
    """Pretty-print the agent result."""
    print("\n" + "="*60)
    print("AGENT RESULT")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Attempts: {result.attempts}")
    print(f"Original Query: {result.original_query}")
    if result.sql:
        print(f"Generated SQL:\n  {result.sql}")
    print(f"Message: {result.final_message}")


# =============================================================================
# Evaluation / Demonstration
# =============================================================================

def run_evaluation():
    """
    Demonstrate the agent with examples including a failure case
    that gets automatically corrected.
    """
    print("\n" + "#"*60)
    print("# EVALUATION: Last-Mile Verification Demonstration")
    print("#"*60)

    # Configure mock LLM with deliberate failure scenarios
    mock_responses = {
        # Case 1: Clean success
        "premium customers": [
            "SELECT name, email FROM customers WHERE tier = 'premium'"
        ],

        # Case 2: Syntax error -> correction
        "total revenue": [
            "SELEC SUM(amount) FROM orders",  # Typo: SELEC
            "SELECT SUM(amount) FROM orders"   # Corrected
        ],

        # Case 3: Schema error -> correction
        "customer purchases": [
            "SELECT name, total_spent FROM customers",  # Wrong column
            "SELECT c.name, SUM(o.amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name"
        ],

        # Case 4: Semantic mismatch -> correction
        "top 5 highest spending": [
            "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",  # Missing ORDER BY, LIMIT
            "SELECT customer_id, SUM(amount) as total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 5"
        ],

        # Case 5: Safety violation -> correction
        "remove all inactive": [
            "DELETE FROM customers",  # Dangerous: no WHERE
            "DELETE FROM customers WHERE tier = 'inactive'",  # Still unsafe pattern
            "SELECT * FROM customers WHERE tier = 'inactive'"  # Converted to safe SELECT
        ],
    }

    llm = MockLLM(responses=mock_responses)
    agent = NLToSQLAgent(llm=llm, max_retries=2)

    test_cases = [
        ("Show me all premium customers", "Clean success"),
        ("What is the total revenue from all orders?", "Syntax error correction"),
        ("List customer purchases by name", "Schema error correction"),
        ("Show the top 5 highest spending customers", "Semantic mismatch correction"),
        ("Remove all inactive customers", "Safety violation handling"),
    ]

    results_summary = []

    for query, description in test_cases:
        print(f"\n{'─'*60}")
        print(f"TEST: {description}")
        print(f"Query: \"{query}\"")
        print("─"*60)

        result = agent.process(query)
        print_result(result)
        print_audit_trail(result.audit_trail)

        results_summary.append({
            "description": description,
            "query": query,
            "success": result.success,
            "attempts": result.attempts,
            "final_sql": result.sql
        })

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Test Case':<35} {'Success':<10} {'Attempts':<10}")
    print("-"*60)
    for r in results_summary:
        print(f"{r['description']:<35} {'✓' if r['success'] else '✗':<10} {r['attempts']:<10}")

    successful = sum(1 for r in results_summary if r['success'])
    print("-"*60)
    print(f"Total: {successful}/{len(results_summary)} successful")

    return results_summary


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_evaluation()
