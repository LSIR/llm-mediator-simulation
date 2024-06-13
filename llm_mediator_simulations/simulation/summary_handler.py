"""Summary class to handle the summary of a conversation"""

from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.utils.model_utils import (
    summarize_conversation_with_last_messages,
)


class SummaryHandler:
    """Summary class to handle the summary of a conversation"""

    def __init__(self, *, model: LanguageModel, latest_messages_limit: int = 3) -> None:
        """Initialize the summary instance.

        Args:
            latest_messages_limit (int, optional): The number of latest messages to keep track of. Defaults to 3.
        """

        self.summary = ""
        self.latest_messages: list[str] = []

        self._model = model
        self._latest_messages_limit = latest_messages_limit

    def update_with_messages(self, messages: list[str]) -> str:
        """Update the summary with the given messages."""

        self.latest_messages = (self.latest_messages + messages)[
            -self._latest_messages_limit :
        ]

        self.summary = summarize_conversation_with_last_messages(
            self._model, self.summary, self.latest_messages
        )

        return self.summary

    def update_with_message(self, message: str) -> str:
        """Update the summary with the given message."""

        return self.update_with_messages([message])
