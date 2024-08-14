"""Async handler class for debate summaries in parallel"""

from typing import override

from llm_mediator_simulation.models.language_model import AsyncLanguageModel
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.utils.interfaces import AsyncPromptable
from llm_mediator_simulation.utils.model_utils import (
    summarize_conversation_with_last_messages_async,
)
from llm_mediator_simulation.utils.types import Intervention


class AsyncSummaryHandler(AsyncPromptable):
    """Summary class to handle the summary of a conversation"""

    def __init__(
        self,
        *,
        config: SummaryConfig,
        model: AsyncLanguageModel,
        parallel_debates: int = 1,
    ) -> None:
        """Initialize the summary instance.

        Args:
            model: The language model to use for the summary.
            latest_messages_limit: The number of latest messages to keep track of. Defaults to 3.
            debaters: The debaters participating in the debate. Defaults to None.
            parallel_debates: The number of parallel debates. Defaults to 1.
        """

        self.summaries = [""] * parallel_debates
        self.latest_messages: list[list[Intervention]] = [
            [] for _ in range(parallel_debates)
        ]

        self._model = model
        self._latest_messages_limit = config.latest_messages_limit

        self.debaters = config.debaters or []
        self.parallel_debates = parallel_debates

    @property
    def message_strings(self) -> list[list[str]]:
        """Return the last message string contents"""

        return [
            [message.text for message in debate if message.text]
            for debate in self.latest_messages
        ]

    def add_new_messages(self, messages: list[Intervention]) -> None:
        """Add new messages to the latest messages list.

        Args:
            messages: The messages to add, 1 per debate. It is assumed that it contains 1 intervention per debate, \
active or not. Empty messages are ignored.
        """

        assert (
            len(messages) == self.parallel_debates
        ), "The number of messages must match the number of debates."

        for index, message in enumerate(messages):
            if not message.text:
                continue

            self.latest_messages[index] = (self.latest_messages[index] + [message])[
                -self._latest_messages_limit :
            ]

    async def regenerate_summaries(self) -> None:
        """Regenerate the debate summaries.
        All summaries are regenerated, even for the individual debates that may not have been updated.
        """

        self.summaries = await summarize_conversation_with_last_messages_async(
            self._model, self.summaries, self.message_strings
        )

    @override
    async def to_prompts(self) -> list[str]:
        msg_sep = "\n\n"

        prompts: list[str] = []

        for messages, summary in zip(self.message_strings, self.summaries):
            prompts.append(
                f"""Here is a summary of the last exchanges (if empty, the conversation just started):
{summary}

Here are the last messages exchanged (you should focus your argumentation on them):
{msg_sep.join(messages)}
"""
            )
        return prompts

    def raw_history_prompts(self) -> list[str]:
        """Return the last messages "as is" in a list of prompts.
        The messages are separated by a newline character."""
        prompts: list[str] = []

        for debate in self.latest_messages:
            prompt_messages: list[str] = []
            for message in debate:
                if message.text:
                    debater_name = (
                        message.debater.name if message.debater else "Mediator"
                    )

                    prompt_messages.append(
                        f"[{message.timestamp}] {debater_name}: {message.text}"
                    )
            prompts.append("\n".join(prompt_messages))

        return prompts

    def debaters_prompt(self) -> str:
        """Returns the debaters that are participating in the experiment"""

        debater_strings: list[str] = []

        for index, debater in enumerate(self.debaters):
            debater_strings.append(f"{debater.name or index}: {debater.name}")

        sep = "\n"

        return f"""PARTICIPANTS
         {sep.join(debater_strings)}"""
