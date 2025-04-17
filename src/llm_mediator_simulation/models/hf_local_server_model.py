"""Mistral local model running as a server wrapper"""

from typing import override

import httpx

from llm_mediator_simulation.models.language_model import LanguageModel


class HFLocalServerModel(LanguageModel):
    """Mistral local model running as a server wrapper
    (to avoid reloading weights before each new debate)."""

    def __init__(self, *, port: int = 8000, max_new_tokens: int = 100) -> None:
        """Initialize a Mistral local model.

        Args:
            port: The port on which the local server is running.
        """

        self.port = port
        self.max_new_tokens = max_new_tokens

    @override
    def sample(self, prompt: str, seed: int | None = None) -> str:
        """Generate text based on the given prompt."""

        try:
            response = httpx.post(
                f"http://localhost:{self.port}/call",
                json={
                    "text": prompt,
                    "seed": seed,
                    "max_new_tokens": self.max_new_tokens,
                },
                timeout=40,
            )
        except httpx.ConnectError:
            return "Local server not running."

        return response.text
