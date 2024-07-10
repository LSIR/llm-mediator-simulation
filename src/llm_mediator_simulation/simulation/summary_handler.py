"""Summary class to handle the summary of a conversation"""

from typing import override

from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.configuration import Debater
from llm_mediator_simulations.utils.interfaces import Promptable
from llm_mediator_simulations.utils.model_utils import (
    summarize_conversation_with_last_messages,
)
from llm_mediator_simulations.utils.types import Intervention


class SummaryHandler(Promptable):
    """Summary class to handle the summary of a conversation"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        latest_messages_limit: int = 3,
        debaters: list[Debater] | None = None,
    ) -> None:
        """Initialize the summary instance.

        Args:
            latest_messages_limit (int, optional): The number of latest messages to keep track of. Defaults to 3.
        """

        self.summary = ""
        self.latest_messages: list[Intervention] = []

        self._model = model
        self._latest_messages_limit = latest_messages_limit

        self.debaters: dict[int, Debater] = {}

        if debaters is not None:
            for index, debater in enumerate(debaters):
                self.debaters[index] = debater

    def update_with_messages(self, messages: list[Intervention]) -> str:
        """Update the summary with the given messages."""

        self.latest_messages = (self.latest_messages + messages)[
            -self._latest_messages_limit :
        ]

        self.summary = summarize_conversation_with_last_messages(
            self._model, self.summary, self.message_strings()
        )

        return self.summary

    def update_with_message(self, message: Intervention) -> str:
        """Update the summary with the given message."""

        return self.update_with_messages([message])

    @override
    def to_prompt(self) -> str:
        msg_sep = "\n\n"

        return f"""Here is a summary of the last exchanges (if empty, the conversation just started):
        {self.summary}

        Here are the last messages exchanged (you should focus your argumentation on them):
        {msg_sep.join(self.message_strings())}
        """

    def message_strings(self) -> list[str]:
        """Return the filtered message strings."""

        strings: list[str] = []

        for message in self.latest_messages:
            if message.text is not None:
                strings.append(message.text)

        return strings

    def raw_history_prompt(self) -> str:
        """Return the last messages "as is"."""
        messages: list[str] = []

        for message in self.latest_messages:
            if message.text is not None and message.authorId is not None:
                debater = self.debaters[message.authorId]
                debater_id = debater.name if debater is not None else message.authorId

                messages.append(f"[{message.timestamp}] {debater_id}: {message.text}")

        return "\n".join(messages)

    def debaters_prompt(self) -> str:
        """Returns the debaters that are participating in the experiment"""

        debater_strings: list[str] = []

        for index, debater in self.debaters.items():
            debater_strings.append(f"{debater.name or index}: {debater.name}")

        sep = "\n"

        return f"""PARTICIPANTS
         {sep.join(debater_strings)}"""
