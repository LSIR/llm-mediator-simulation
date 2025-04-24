"""Mistral local model running as a server wrapper"""

from typing import Any, override

import httpx

from llm_mediator_simulation.models.language_model import LanguageModel


class HFLocalServerModel(LanguageModel):
    """Mistral local model running as a server wrapper
    (to avoid reloading weights before each new debate)."""

    def __init__(self, *, port: int = 8000, **kwargs: Any) -> None:
        """Initialize a Mistral local model.

        Args:
            port: The port on which the local server is running.
            kwargs: Additional arguments for the model.
        """

        self.port = port
        for key, value in kwargs.items():
            setattr(self, key, value)

    @override
    def sample(self, prompt: str, seed: int | None = None, **kwargs: Any) -> str:
        """Generate text based on the given prompt."""
        data = {
            "text": prompt,
            "seed": seed,
        }

        for key, value in kwargs.items():
            assert key not in data
            data[key] = value

        # get all parameters from self
        for parameter in self.__dict__.keys():
            if parameter not in ["port"] and parameter not in kwargs:
                data[parameter] = getattr(self, parameter)

        try:
            response = httpx.post(
                f"http://localhost:{self.port}/call",
                json=data,  # {"text": prompt, "seed": seed},
                timeout=80,
            )
        except httpx.ConnectError:
            return "Local server not running."

        return response.text
