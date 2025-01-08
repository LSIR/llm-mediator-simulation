"""Abstract base class for language models."""

from abc import ABC, abstractmethod


class LanguageModel(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def sample(self, prompt: str) -> str:
        """Generate text based on the given prompt."""
        
    @abstractmethod
    def generate_response(self, prompt:str) -> str:
        """Generate text based on the given prompt, without returning the prompt."""
    


class AsyncLanguageModel(ABC):
    """Abstract base class for async language models."""

    @abstractmethod
    async def sample(self, prompts: list[str]) -> list[str]:
        """Generate texts based on the given prompts."""
