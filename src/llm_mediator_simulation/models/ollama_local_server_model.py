"""Ollama local model running as a server wrapper
BUG: Seeding with Ollama is not well supported as of April 04th 2025: https://github.com/ollama/ollama/issues/5321
likely due to KV Caching https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

"""

import asyncio
from typing import Any, override

import ollama

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)


class OllamaLocalModel(LanguageModel):
    """Ollama local model running as a server wrapper"""

    def __init__(self, *, model_name: str = "deepseek-r1:8b") -> None:
        """Initialize a Ollama local model.

        Args:
            model_name: The model name to use.
        """

        self.model_name = model_name

    @override
    def sample(self, prompt: str, seed: int | None = None, **kwargs: Any) -> str:
        """Generate text based on the given prompt."""

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"seed": seed},
        )
        return response.response


class AsyncOllamaLocalModel(AsyncLanguageModel):
    """Asynchronous Ollama local model running as a server wrapper"""

    def __init__(self, *, model_name: str = "deepseek-r1:8b") -> None:
        """Initialize a Ollama local model.

        Args:
            model_name: The model name to use.
        """

        self.model_name = model_name
        self.client = ollama.AsyncClient()

    @override
    async def sample(
        self, prompts: list[str], seed: int | None = None, **kwargs: Any
    ) -> list[str]:
        """Generate text based on the given prompt."""

        # Await all completions asynchronously
        results = await asyncio.gather(
            *[
                self.client.generate(
                    model=self.model_name, prompt=prompt, options={"seed": seed}
                )
                for prompt in prompts
            ]
        )
        return [response.response for response in results]
