"""
LangChain Tool Adapter
======================
Wraps the verified NL-to-SQL agent as a LangChain Tool.

This is a thin wrapper - all verification and correction logic
remains in the core agent. LangChain only provides the interface.

Requirements:
    pip install langchain-core
"""

from typing import Optional, Type

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nl_to_sql_agent import NLToSQLAgent, LLMInterface, AgentResult

try:
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True

    class VerifiedSQLToolInput(BaseModel):
        """Input schema for the VerifiedSQLTool."""
        query: str = Field(
            description="Natural language question to convert to verified SQL"
        )
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    VerifiedSQLToolInput = None


class VerifiedSQLTool(BaseTool):
    """
    LangChain Tool wrapper for the verified NL-to-SQL agent.

    This tool exposes the full verification pipeline to LangChain agents.
    All verification, correction, and audit logging happens in the core agent.

    Example:
        ```python
        from nl_to_sql_agent import NLToSQLAgent, MockLLM
        from adapters.langchain_tool import VerifiedSQLTool

        llm = MockLLM(responses={"customers": ["SELECT * FROM customers"]})
        agent = NLToSQLAgent(llm=llm)
        tool = VerifiedSQLTool(agent=agent)

        result = tool.invoke({"query": "Show all customers"})
        ```
    """

    name: str = "verified_sql_generator"
    description: str = (
        "Converts natural language questions into verified SQL queries. "
        "The SQL is validated for syntax, schema correctness, safety, and "
        "semantic alignment before being returned. Use this when you need "
        "to generate reliable SQL from user questions."
    )
    args_schema: Type[BaseModel] = VerifiedSQLToolInput
    return_direct: bool = False

    # Core agent instance (not a Pydantic field)
    agent: NLToSQLAgent = None
    include_audit: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, agent: NLToSQLAgent, include_audit: bool = False, **kwargs):
        """
        Initialize the tool with a configured NLToSQLAgent.

        Args:
            agent: Pre-configured NLToSQLAgent instance
            include_audit: If True, include audit trail in output
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this adapter. "
                "Install with: pip install langchain-core"
            )
        super().__init__(**kwargs)
        self.agent = agent
        self.include_audit = include_audit

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Execute the verified SQL generation.

        Args:
            query: Natural language question
            run_manager: LangChain callback manager (optional)

        Returns:
            Verified SQL query or error message
        """
        result: AgentResult = self.agent.process(query)

        if result.success:
            output = f"SQL: {result.sql}"
            if self.include_audit:
                output += f"\n\nVerification: Passed after {result.attempts} attempt(s)"
            return output
        else:
            return f"Error: {result.final_message}"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version - delegates to sync for simplicity in V0."""
        return self._run(query, run_manager)

    def get_last_audit_trail(self) -> list:
        """
        Retrieve the audit trail from the last query.

        This provides full traceability for compliance/debugging.
        """
        return self.agent.audit_trail


# Convenience function for quick setup
def create_verified_sql_tool(
    llm: LLMInterface,
    max_retries: int = 3,
    include_audit: bool = False
) -> VerifiedSQLTool:
    """
    Factory function to create a VerifiedSQLTool with a new agent.

    Args:
        llm: LLM interface implementation
        max_retries: Maximum correction attempts
        include_audit: Include audit info in output

    Returns:
        Configured VerifiedSQLTool instance
    """
    agent = NLToSQLAgent(llm=llm, max_retries=max_retries)
    return VerifiedSQLTool(agent=agent, include_audit=include_audit)
