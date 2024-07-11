"""Generic utility interfaces for common behavior"""

from abc import ABC, abstractmethod


class Promptable(ABC):
    """Interface for classes that can be transformed into a prompt."""

    @abstractmethod
    def to_prompt(self) -> str:
        """Transform the instance into a prompt."""


class AsyncPromptable(ABC):
    """Interface for classes that can be transformed into a prompt, in a async / batched way."""

    @abstractmethod
    async def to_prompts(self) -> list[str]:
        """Transform the instance into a list of prompts."""
