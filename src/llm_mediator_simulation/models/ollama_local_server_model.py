"""Ollama local model running as a server wrapper"""

from typing import override
from llm_mediator_simulation.models.language_model import LanguageModel
import ollama


class OllamaLocalModel(LanguageModel):
    """Ollama local model running as a server wrapper"""

    def __init__(self, *, model_name: str = "deepseek-r1:8b") -> None:
        """Initialize a Ollama local model.

        Args:
            model_name: The model name to use.
        """

        self.model_name = model_name

    @override
    def sample(self, prompt: str, seed: int | None = None) -> str:
        """Generate text based on the given prompt."""

        response = ollama.generate(
            model=self.model_name, prompt=prompt, options={"seed": seed}
        )
        return response.response
