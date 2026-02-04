"""
Semantic Verifier
=================

Validates that the SQL appears to match the user's intent.
"""

from nl_to_sql.models import VerificationResult, VerificationStatus
from nl_to_sql.verifiers.base import Verifier


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
        """
        Verify SQL semantically matches the original query intent.

        Args:
            sql: SQL query to validate
            context: Must contain 'original_query' key

        Returns:
            VerificationResult with PASSED or FAILED status
        """
        original_query = context.get("original_query", "").lower()
        sql_lower = sql.lower()

        issues = []

        # Check for aggregation keywords
        agg_keywords = ["total", "sum", "count", "average", "avg", "how many"]
        has_agg_intent = any(kw in original_query for kw in agg_keywords)
        has_agg_sql = any(
            fn in sql_lower for fn in ["count(", "sum(", "avg(", "max(", "min("]
        )

        if has_agg_intent and not has_agg_sql:
            issues.append("Query seems to request aggregation but SQL has none")

        # Check for ordering keywords
        order_keywords = [
            "highest",
            "lowest",
            "top",
            "bottom",
            "most",
            "least",
            "largest",
            "smallest",
        ]
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
                details={"issues": issues},
            )

        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.PASSED,
            message="SQL appears to match query intent",
        )
