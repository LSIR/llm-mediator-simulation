"""Abstract base class for language models."""

from abc import ABC, abstractmethod


class LanguageModel(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def sample(self, prompt: str) -> str:
        """Generate text based on the given prompt."""
