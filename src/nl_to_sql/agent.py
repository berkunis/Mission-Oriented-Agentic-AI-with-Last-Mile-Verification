"""
NL-to-SQL Agent Controller
==========================

Main agent that orchestrates NL-to-SQL conversion with verification.
"""

from datetime import datetime

from nl_to_sql.llm.base import LLMInterface
from nl_to_sql.models import AgentResult, AuditEntry, VerificationResult, VerificationStatus
from nl_to_sql.verifiers.base import VerificationChain
from nl_to_sql.verifiers.syntax import SAMPLE_SCHEMA


class NLToSQLAgent:
    """
    Main agent that orchestrates NL-to-SQL conversion with verification.

    The agent:
    1. Takes a natural language query
    2. Generates SQL using the configured LLM
    3. Verifies the SQL through the verification chain
    4. Automatically retries with correction prompts on failure
    5. Maintains a full audit trail of all operations
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
        verification_chain: VerificationChain | None = None,
        max_retries: int = 3,
        schema: dict | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            llm: LLM interface for SQL generation
            verification_chain: Chain of verifiers (defaults to standard chain)
            max_retries: Maximum number of correction attempts
            schema: Database schema for validation
        """
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
        verification_results: list[VerificationResult] | None = None,
    ) -> None:
        """Add entry to audit trail."""
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            step=step,
            input_data=input_data,
            output_data=output_data,
            verification_results=verification_results or [],
        )
        self.audit_trail.append(entry)

    def _extract_sql(self, llm_output: str) -> str:
        """Extract SQL from LLM output, handling markdown code blocks."""
        sql = llm_output.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            # Remove first and last lines (code block markers)
            sql = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return sql.strip()

    def process(self, natural_language_query: str) -> AgentResult:
        """
        Main entry point: convert NL query to verified SQL.

        Args:
            natural_language_query: The user's question in natural language

        Returns:
            AgentResult with success status, SQL (if successful), and audit trail
        """
        self.audit_trail = []  # Reset for new query

        context = {
            "original_query": natural_language_query,
            "schema": self.schema,
        }

        # Initial generation
        prompt = f"{self.SYSTEM_PROMPT}\n\nQuestion: {natural_language_query}"

        for attempt in range(self.max_retries + 1):
            # Generate SQL
            self._log_audit(
                step=f"generation_attempt_{attempt + 1}",
                input_data={
                    "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt
                },
                output_data={},
            )

            response = self.llm.generate(prompt)
            sql = self._extract_sql(response.content)

            self._log_audit(
                step=f"llm_response_{attempt + 1}",
                input_data={},
                output_data={"sql": sql, "model": response.model},
            )

            # Verify
            passed, verification_results = self.verification_chain.run(sql, context)

            self._log_audit(
                step=f"verification_{attempt + 1}",
                input_data={"sql": sql},
                output_data={"passed": passed},
                verification_results=verification_results,
            )

            if passed:
                return AgentResult(
                    success=True,
                    sql=sql,
                    original_query=natural_language_query,
                    attempts=attempt + 1,
                    audit_trail=self.audit_trail,
                    final_message=f"Query verified successfully after {attempt + 1} attempt(s)",
                )

            # Prepare correction prompt for next attempt
            failed_result = next(
                r for r in verification_results if r.status == VerificationStatus.FAILED
            )

            prompt = self.CORRECTION_PROMPT_TEMPLATE.format(
                original_query=natural_language_query,
                previous_sql=sql,
                error_message=failed_result.message,
            )

        # Max retries exceeded
        return AgentResult(
            success=False,
            sql=None,
            original_query=natural_language_query,
            attempts=self.max_retries + 1,
            audit_trail=self.audit_trail,
            final_message=f"Failed to generate valid SQL after {self.max_retries + 1} attempts",
        )
