"""
Base LLM Interface
==================

Abstract interface for LLM providers.
"""

from abc import ABC, abstractmethod

from nl_to_sql.models import LLMResponse


class LLMInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt for context

        Returns:
            LLMResponse with generated content
        """
        pass
