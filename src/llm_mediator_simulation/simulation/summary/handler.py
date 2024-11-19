"""Handler class for summaries"""

from typing import override

from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.utils.interfaces import Promptable
from llm_mediator_simulation.utils.model_utils import (
    summarize_conversation_with_last_messages,
)
from llm_mediator_simulation.utils.types import Intervention
import datetime


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

        #self.time_indicator = 0 
        self.latest_messages: list[Intervention] = []
        self._model = model
        self._latest_messages_limit = config.latest_messages_limit

        self.debaters = config.debaters or []
        

    @property
    def message_strings(self) -> list[str]:
        """Return the last message string contents"""

        return [message.text for message in self.latest_messages if message.text]
    
    @property
    def message_speakers_and_strings(self) -> list[str]:
        """Return the name of last messages"""
        return [f"{message.debater.name} said:  \"{message.text} \"" if message.debater is not None else f"Mediator: {message.text}" for message in self.latest_messages if message.text]
    

    def add_new_message(self, message: Intervention) -> None:
        """Add a new message to the latest messages list.
        Empty messages are ignored."""

        if not message.text:
            return

        self.latest_messages = (self.latest_messages + [message])[
            -self._latest_messages_limit :
        ]
        #self.time_indicator += 1

    def regenerate_summary(self) -> str:
        """Regenerate the summary with the latest messages."""

        self.summary = summarize_conversation_with_last_messages(
            self._model, self.summary, self.message_speakers_and_strings
        )

        return self.summary

    @override
    def to_prompt(self) -> str:
        msg_sep = "\n\n"

        if len(self.latest_messages) == 0:
            return "The conversation has just started, and there are no prior messages or exchanges. Please present your initial argument on the topic"
        return f"""Here is a summary of the last exchanges :
        {self.summary}
        Here are the last messages exchanged (you should focus your argumentation on them):
        {msg_sep.join(self.message_speakers_and_strings)}
        """

    def raw_history_prompt(self) -> str:
        """Return the last messages "as is"."""
        messages: list[str] = []

        for message in self.latest_messages:
            if message.text:
                debater_name = message.debater.name if message.debater else "Mediator"

                messages.append(f"[Time:{message.timestamp.minute}:{message.timestamp.second}] {debater_name}: {message.text}")

        return "\n".join(messages)

    def debaters_prompt(self) -> str:
        """Returns the debaters that are participating in the experiment"""

        debater_strings: list[str] = []

        for index, debater in enumerate(self.debaters):
            debater_strings.append(f"{debater.name or index}: {debater.name}")

        sep = "\n"

        return f"""PARTICIPANTS
         {sep.join(debater_strings)}"""
