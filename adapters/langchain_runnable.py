"""
LangChain Runnable (LCEL) Adapter
=================================
Exposes the verified NL-to-SQL agent as a LangChain Runnable.

This allows the agent to be composed in LCEL chains while preserving
the internal verification loop and audit trail.

Requirements:
    pip install langchain-core
"""

from typing import Any, Dict, Iterator, Optional, Union
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl_to_sql_agent import NLToSQLAgent, AgentResult, LLMInterface

try:
    from langchain_core.runnables import Runnable, RunnableConfig
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Runnable = object
    RunnableConfig = None


class VerifiedSQLRunnable(Runnable):
    """
    LangChain Runnable exposing the verified NL-to-SQL agent.

    This Runnable wraps the complete verification loop, making it
    composable in LCEL chains. The internal agent logic is unchanged.

    Input: str (natural language query) or dict with "query" key
    Output: AgentResult (full result) or str (SQL only, if output_sql_only=True)

    Example:
        ```python
        from nl_to_sql_agent import NLToSQLAgent, MockLLM
        from adapters.langchain_runnable import VerifiedSQLRunnable

        llm = MockLLM(responses={"customers": ["SELECT * FROM customers"]})
        agent = NLToSQLAgent(llm=llm)

        runnable = VerifiedSQLRunnable(agent=agent)

        # Use directly
        result = runnable.invoke("Show all customers")

        # Compose in LCEL chain
        from langchain_core.runnables import RunnablePassthrough
        chain = RunnablePassthrough() | runnable
        result = chain.invoke("Show all customers")
        ```
    """

    def __init__(
        self,
        agent: NLToSQLAgent,
        output_sql_only: bool = False,
        raise_on_failure: bool = False
    ):
        """
        Initialize the Runnable.

        Args:
            agent: Pre-configured NLToSQLAgent instance
            output_sql_only: If True, return only SQL string (or None on failure)
            raise_on_failure: If True, raise ValueError on verification failure
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this adapter. "
                "Install with: pip install langchain-core"
            )
        self.agent = agent
        self.output_sql_only = output_sql_only
        self.raise_on_failure = raise_on_failure

    @property
    def InputType(self) -> type:
        return Union[str, Dict[str, Any]]

    @property
    def OutputType(self) -> type:
        return str if self.output_sql_only else AgentResult

    def _extract_query(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Extract query string from various input formats."""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            # Support multiple key conventions
            for key in ["query", "question", "input", "text"]:
                if key in input_data:
                    return input_data[key]
            raise ValueError(
                f"Dict input must contain one of: query, question, input, text. "
                f"Got keys: {list(input_data.keys())}"
            )
        else:
            raise TypeError(f"Expected str or dict, got {type(input_data)}")

    def invoke(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Union[AgentResult, str, None]:
        """
        Execute the verified SQL generation.

        Args:
            input: Natural language query (str or dict with "query" key)
            config: LangChain runnable config (optional)

        Returns:
            AgentResult (full) or str (SQL only) depending on output_sql_only
        """
        query = self._extract_query(input)
        result: AgentResult = self.agent.process(query)

        if not result.success and self.raise_on_failure:
            raise ValueError(
                f"SQL verification failed after {result.attempts} attempts: "
                f"{result.final_message}"
            )

        if self.output_sql_only:
            return result.sql  # None if failed
        return result

    async def ainvoke(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Union[AgentResult, str, None]:
        """Async version - delegates to sync for simplicity in V0."""
        return self.invoke(input, config, **kwargs)

    def stream(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream verification progress as it happens.

        Yields audit entries as they are created, providing
        real-time visibility into the verification process.
        """
        query = self._extract_query(input)

        # Process and yield audit entries
        result = self.agent.process(query)

        for entry in result.audit_trail:
            yield {
                "type": "audit_entry",
                "step": entry.step,
                "timestamp": entry.timestamp,
                "data": entry.output_data,
                "verifications": [
                    {"name": vr.verifier_name, "status": vr.status.value, "message": vr.message}
                    for vr in entry.verification_results
                ] if entry.verification_results else None
            }

        # Final result
        yield {
            "type": "final_result",
            "success": result.success,
            "sql": result.sql,
            "attempts": result.attempts,
            "message": result.final_message
        }

    def get_audit_trail(self) -> list:
        """Retrieve the audit trail from the last invocation."""
        return self.agent.audit_trail


# Convenience factory
def create_verified_sql_runnable(
    llm: LLMInterface,
    max_retries: int = 3,
    output_sql_only: bool = False,
    raise_on_failure: bool = False
) -> VerifiedSQLRunnable:
    """
    Factory function to create a VerifiedSQLRunnable with a new agent.

    Args:
        llm: LLM interface implementation
        max_retries: Maximum correction attempts
        output_sql_only: Return only SQL string
        raise_on_failure: Raise on verification failure

    Returns:
        Configured VerifiedSQLRunnable instance
    """
    agent = NLToSQLAgent(llm=llm, max_retries=max_retries)
    return VerifiedSQLRunnable(
        agent=agent,
        output_sql_only=output_sql_only,
        raise_on_failure=raise_on_failure
    )
