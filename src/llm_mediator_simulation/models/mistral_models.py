"""Mistral model wrapper."""  # TODO Likely deprecated. Remove or update.

from typing import Literal, override

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from llm_mediator_simulation.models.language_model import LanguageModel


class MistralModel(LanguageModel):
    """Mistral model wrapper."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: Literal["mistral-large-latest", "mistral-small-latest"],
    ):
        """Initialize a Mistral model.

        Args:
            api_key: OpenAI API key.
            model_name: OpenAI model name.
        """

        self.client = MistralClient(api_key=api_key)
        self.model_name = model_name

    @override
    def sample(self, prompt: str, seed: int | None = None) -> str:
        """Generate text based on the given prompt."""

        response = self.client.chat(
            model=self.model_name, messages=[ChatMessage(role="user", content=prompt)]
        )

        return response.choices[0].message.content
