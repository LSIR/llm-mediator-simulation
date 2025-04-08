"""Handler class for summaries"""

from typing import override

from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.utils.interfaces import Promptable
from llm_mediator_simulation.utils.model_utils import (
    summarize_conversation_with_last_messages,
)
from llm_mediator_simulation.utils.types import Intervention


class SummaryHandler(Promptable):
    """Summary class to handle the summary of a conversation"""

    def __init__(
        self,
        *,
        config: SummaryConfig,
        model: LanguageModel,
    ) -> None:
        """Initialize the summary instance.

        Args:
            latest_messages_limit (int, optional): The number of latest messages to keep track of. Defaults to 3.
        """

        self.summary = ""
        self.latest_messages: list[Intervention] = []

        self._model = model
        self._latest_messages_limit = config.latest_messages_limit

        self.debaters = config.debaters or []
        self.ignore = config.ignore
        self.utterance = config.utterance

    @property
    def message_strings(self) -> list[str]:
        """Return the last message string contents"""
        message_list = []
        for message in self.latest_messages:
            if message.text:
                if message.debater:
                    author_name = message.debater.name
                else:
                    author_name = "Mediator"
                message_list.append(f"""- {author_name}: "{message.text}\"""")

        return message_list

    def add_new_message(self, message: Intervention) -> None:
        """Add a new message to the latest messages list.
        Empty messages are ignored."""

        if not message.text:
            return

        self.latest_messages = (self.latest_messages + [message])[
            -self._latest_messages_limit :
        ]

    def regenerate_summary(self, seed: int | None = None) -> str:
        """Regenerate the summary with the latest messages."""
        if self.ignore:
            pass
        else:
            self.summary = summarize_conversation_with_last_messages(
                self._model, self.summary, self.message_strings, seed
            )

        return self.summary

    @override
    def to_prompt(self) -> str:
        msg_sep = "\n\n"

        if not self.message_strings:
            return ""

        prompt = ""

        if not self.ignore:
            prompt += f"""Here is a summary of the conversation so far:
{self.summary}\n\n"""  # TODO Add personalized summary... "According to you, here is a summary..."

        prompt += f"""Here are the last {self.utterance}s:
{msg_sep.join(self.message_strings)}
"""

        return prompt

    def raw_history_prompt(self) -> str:
        """Return the last messages "as is"."""
        messages: list[str] = []

        for message in self.latest_messages:
            if message.text:
                debater_name = message.debater.name if message.debater else "Mediator"

                messages.append(f"[{message.timestamp}] {debater_name}: {message.text}")

        return "\n".join(messages)

    def debaters_prompt(self) -> str:
        """Returns the debaters that are participating in the experiment"""

        debater_strings: list[str] = []

        for index, debater in enumerate(self.debaters):
            debater_strings.append(f"{debater.name or index}: {debater.name}")

        sep = "\n"

        return f"""PARTICIPANTS
         {sep.join(debater_strings)}"""
