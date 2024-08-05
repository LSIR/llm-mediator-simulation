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

    def add_new_messages(self, messages: list[Intervention]) -> None:
        """Add new messages to the latest messages list."""

        self.latest_messages = (self.latest_messages + messages)[
            -self._latest_messages_limit :
        ]

    def add_new_message(self, message: Intervention) -> None:
        """Add a new message to the latest messages list."""

        self.add_new_messages([message])

    def regenerate_summary(self) -> str:
        """Regenerate the summary with the latest messages."""

        self.summary = summarize_conversation_with_last_messages(
            self._model, self.summary, self.message_strings()
        )

        return self.summary

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

    def add_new_message(
        self, messages: list[Intervention], active: list[int] | None = None
    ) -> None:
        """Add new messages to the latest messages list.

        Args:
            messages: The messages to add, 1 per debate. It is assumed that it contains 1 intervention per debate, \
                active or not.
            active: The indices of the debates that need to be updated. It is used to filter the messages.
        """

        if active is None:
            active = list(range(self.parallel_debates))

        assert (
            len(messages) == self.parallel_debates
        ), "The number of messages must match the number of debates."

        for index, message in enumerate(messages):
            if index not in active:
                continue

            self.latest_messages[index] = (self.latest_messages[index] + [message])[
                -self._latest_messages_limit :
            ]

    def add_new_messages(
        self, messages: list[list[Intervention]], active: list[int] | None = None
    ) -> None:
        """Add new messages to the latest messages list.

        Args:
            messages: The messages to add, 1 list of them per debate. It is assumed that it contains 1 list per debate, \
                active or not.
            active: The indices of the debates that need to be updated. It is used to filter the messages.
        """

        if active is None:
            active = list(range(self.parallel_debates))

        assert (
            len(messages) == self.parallel_debates
        ), "The number of messages must match the number of debates."

        for index, message_list in enumerate(messages):
            if index not in active:
                continue

            self.latest_messages[index] = (self.latest_messages[index] + message_list)[
                -self._latest_messages_limit :
            ]

    async def regenerate_summaries(self, active: list[int] | None = None) -> None:
        """Regenerate the debate summaries. If `active` is set, only the debate with the given
        indices will have their summaries updated."""

        summaries = self.summaries

        if active:
            summaries = [self.summaries[i] for i in active]

        summaries = await summarize_conversation_with_last_messages_async(
            self._model, summaries, self.message_strings(active)
        )

        if active:
            for i, summary in zip(active, summaries):
                self.summaries[i] = summary
        else:
            self.summaries = summaries

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

    def message_strings(self, active: list[int] | None = None) -> list[list[str]]:
        """Return the filtered message strings.

        Args:
            active: The indices of the debates to consider. If None, all debates are considered.
        """

        debates = self.latest_messages

        if active:
            debates = [self.latest_messages[i] for i in active]

        return [
            [message.text for message in debate if message.text is not None]
            for debate in debates
        ]

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
