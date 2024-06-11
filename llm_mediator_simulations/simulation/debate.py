"""Online debate simulation handler class"""

from dataclasses import dataclass
from enum import Enum

from rich.progress import track

from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.summary import Summary


class DebatePosition(Enum):
    """Debate positions for the participants."""

    AGAINST = 0
    FOR = 1


class Personality(Enum):
    """Debater personality qualifiers."""

    # Mood
    ANGRY = "angry"
    AGGRESSIVE = "aggressive"
    CALM = "calm"
    INSULTING = "insulting"

    # Political
    CONSERVATIVE = "conservative"
    LIBERAL = "liberal"
    LIBERTARIAN = "libertarian"


@dataclass
class Debater:
    """Debater metadata class

    Args:
        position (DebatePosition): The position of the debater.
        personality (str | None, optional): The personality of the debater (as a list of qualifiers). Defaults to None.
    """

    position: DebatePosition
    personality: list[Personality] | None = None


class Debate:
    """Online debate simulation handler class"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        topic: str,
        debaters: list[Debater],
        context="You are taking part in an online debate about the following topic:",
        instructions="Answer with short chat messages (ranging from one to three sentences maximum). You must convince the general public of your position.",
        summary_handler: Summary | None = None,
    ) -> None:
        """Initialize the debate instance.

        Args:
            model (LanguageModel): The language model to use.
            topic (str): The topic of the debate.
            debaters (list[Debater]): The debaters participating in the debate.
            context (str, optional): The context of the debate.
            instructions (str, optional): The instructions for the debate and how to answer.
            summary_handler (Summary | None, optional): The summary handler to use. Defaults to None.
        """

        # Prompt context and metadata
        self.context = context
        self.topic = topic
        self.instructions = instructions

        # Positions
        self.prompt_for = "You are arguing in favor of the statement."
        self.prompt_against = "You are arguing against the statement."

        # Debaters
        self.debaters = debaters

        # Logs
        self.messages: list[tuple[DebatePosition, str]] = []

        self.model = model

        if summary_handler is None:
            self.summary_handler = Summary(model=model)
        else:
            self.summary_handler = summary_handler

    def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.
        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """

        for _ in track(range(rounds), total=rounds):
            for debater in self.debaters:

                # Prepare the prompt.
                prompt = f"""{self.context} {self.topic}. {self.prompt_for if debater.position == DebatePosition.FOR else self.prompt_against}
                {self.instructions}
                
                Your personality is {', '.join(map(lambda x: x.value, debater.personality or []))}.

                Here is a summary of the last exchanges (if empty, the conversation just started):
                {self.summary_handler.summary}

                Here are the last messages exchanged (you should focus your argumentation on them):
                {'\n\n'.join(self.summary_handler.latest_messages)}
                """

                message = self.model.sample(prompt)
                self.messages.append((debater.position, message))

                self.summary_handler.update_with_message(message)
