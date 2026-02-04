"""
LangChain Adapters (Optional)
=============================
Thin wrappers exposing the verified NL-to-SQL agent to LangChain ecosystems.

These adapters are OPTIONAL - the core agent runs without LangChain installed.
LangChain is only imported when these adapters are explicitly used.
"""

# Lazy imports to keep LangChain optional
def get_tool_adapter():
    """Get the LangChain Tool adapter (requires langchain-core)."""
    from .langchain_tool import VerifiedSQLTool
    return VerifiedSQLTool


def get_runnable_adapter():
    """Get the LangChain Runnable adapter (requires langchain-core)."""
    from .langchain_runnable import VerifiedSQLRunnable
    return VerifiedSQLRunnable


__all__ = ["get_tool_adapter", "get_runnable_adapter"]
