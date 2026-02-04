"""
LLM Module
==========

Pluggable LLM interfaces for SQL generation.
"""

from nl_to_sql.llm.base import LLMInterface
from nl_to_sql.llm.mock import MockLLM

__all__ = [
    "LLMInterface",
    "MockLLM",
]
