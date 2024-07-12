"""Summary class to handle the summary of a conversation"""

from typing import override

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from llm_mediator_simulation.simulation.configuration import Debater
from llm_mediator_simulation.utils.interfaces import AsyncPromptable, Promptable
from llm_mediator_simulation.utils.model_utils import (
    summarize_conversation_with_last_messages,
    summarize_conversation_with_last_messages_async,
)
from llm_mediator_simulation.utils.types import Intervention


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

        self.debaters = debaters or []

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
                debater_id = (
                    self.debaters[message.authorId].name
                    if message.authorId in self.debaters
                    else message.authorId
                )

                messages.append(f"[{message.timestamp}] {debater_id}: {message.text}")

        return "\n".join(messages)

    def debaters_prompt(self) -> str:
        """Returns the debaters that are participating in the experiment"""

        debater_strings: list[str] = []

        for index, debater in enumerate(self.debaters):
            debater_strings.append(f"{debater.name or index}: {debater.name}")

        sep = "\n"

        return f"""PARTICIPANTS
         {sep.join(debater_strings)}"""


class AsyncSummaryHandler(AsyncPromptable):
    """Summary class to handle the summary of a conversation"""

    def __init__(
        self,
        *,
        model: AsyncLanguageModel,
        latest_messages_limit: int = 3,
        debaters: list[Debater] | None = None,
        parallel_debates: int = 1,
    ) -> None:
        """Initialize the summary instance.

        Args:
            model: The language model to use for the summary.
            latest_messages_limit: The number of latest messages to keep track of. Defaults to 3.
            debaters: The debaters participating in the debate. Defaults to None.
            parallel_debates: The number of parallel debates. Defaults to 1.
        """

        self.summaries = ["" for _ in range(parallel_debates)]
        self.latest_messages: list[list[Intervention]] = [
            [] for _ in range(parallel_debates)
        ]

        self._model = model
        self._latest_messages_limit = latest_messages_limit

        self.debaters = debaters or []
        self.parallel_debates = parallel_debates

    async def update_with_messages(
        self, messages: list[list[Intervention]]
    ) -> list[str]:
        """Update the summaries with the given messages."""

        # 1. Update the latest messages
        self.latest_messages = (self.latest_messages + messages)[
            -self._latest_messages_limit :
        ]

        self.summaries = await summarize_conversation_with_last_messages_async(
            self._model, self.summaries, self.message_strings()
        )

        return self.summaries

    async def update_with_message(self, message: list[Intervention]) -> list[str]:
        """Update every summary with one additional message each."""

        return await self.update_with_messages([message])

    @override
    async def to_prompts(self) -> list[str]:
        msg_sep = "\n\n"

        prompts: list[str] = []

        for messages, summary in zip(self.message_strings(), self.summaries):
            prompts.append(
                f"""Here is a summary of the last exchanges (if empty, the conversation just started):
            {summary}

            Here are the last messages exchanged (you should focus your argumentation on them):
            {msg_sep.join(messages)}
            """
            )
        return prompts

    def message_strings(self) -> list[list[str]]:
        """Return the filtered message strings."""

        return [
            [message.text for message in debate if message.text is not None]
            for debate in self.latest_messages
        ]

    def raw_history_prompts(self) -> list[str]:
        """Return the last messages "as is" in a list of prompts.
        The messages are separated by a newline character."""
        prompts: list[str] = []

        for debate in self.latest_messages:
            prompt_messages: list[str] = []
            for message in debate:
                if message.text is not None and message.authorId is not None:
                    debater_id = (
                        self.debaters[message.authorId].name
                        if message.authorId in self.debaters
                        else message.authorId
                    )

                    prompt_messages.append(
                        f"[{message.timestamp}] {debater_id}: {message.text}"
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
