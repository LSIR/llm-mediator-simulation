"""Abstract base class for language models."""

from abc import ABC, abstractmethod


class LanguageModel(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def sample(self, prompt: str, seed: int | None = None) -> str:
        """Generate text based on the given prompt."""


class AsyncLanguageModel(ABC):
    """Abstract base class for async language models."""

    @abstractmethod
    async def sample(self, prompts: list[str], seed: int | None = None) -> list[str]:
        """Generate texts based on the given prompts."""
