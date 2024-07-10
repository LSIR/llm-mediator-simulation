"""Generic utility interfaces for common behavior"""

from abc import ABC, abstractmethod


class Promptable(ABC):
    """Interface for classes that can be transformed into a prompt."""

    @abstractmethod
    def to_prompt(self) -> str:
        """Transform the instance into a prompt."""
