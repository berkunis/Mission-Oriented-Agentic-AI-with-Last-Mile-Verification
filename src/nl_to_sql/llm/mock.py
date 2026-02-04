"""
Mock LLM
========

Mock LLM implementation for testing and demonstration.
"""

from nl_to_sql.llm.base import LLMInterface
from nl_to_sql.models import LLMResponse


class MockLLM(LLMInterface):
    """
    Mock LLM for demonstration and testing purposes.

    In production, replace with OpenAI, Anthropic, or other provider.
    """

    def __init__(self, responses: dict[str, list[str]] | None = None) -> None:
        """
        Initialize with canned responses.

        Args:
            responses: Dict mapping query substrings to list of SQL attempts.
                       Each attempt is returned in sequence (for testing correction).
        """
        self.responses = responses or {}
        self.call_counts: dict[str, int] = {}

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """
        Generate a mock SQL response.

        Matches prompt against configured responses and returns
        successive attempts to simulate correction behavior.
        """
        # Find matching response based on prompt content
        for key, sql_attempts in self.responses.items():
            if key.lower() in prompt.lower():
                count = self.call_counts.get(key, 0)
                self.call_counts[key] = count + 1

                # Return successive attempts (simulating correction)
                attempt_idx = min(count, len(sql_attempts) - 1)
                return LLMResponse(
                    content=sql_attempts[attempt_idx],
                    model="mock-llm-v1",
                )

        # Default fallback
        return LLMResponse(
            content="SELECT * FROM unknown_table",
            model="mock-llm-v1",
        )

    def reset(self) -> None:
        """Reset call counts for fresh test runs."""
        self.call_counts = {}
