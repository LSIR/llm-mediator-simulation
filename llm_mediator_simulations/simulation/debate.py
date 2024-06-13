"""Online debate simulation handler class"""

import pickle
from datetime import datetime
from typing import TypedDict

from rich.progress import track

from llm_mediator_simulations.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.configuration import (
    DebateConfig,
    Debater,
    Mediator,
)
from llm_mediator_simulations.simulation.prompt import (
    debater_comment,
    mediator_comment,
    should_mediator_intervene,
    should_participant_intervene,
)
from llm_mediator_simulations.simulation.summary_handler import SummaryHandler
from llm_mediator_simulations.utils.types import Message


class Debate:
    """Online debate simulation handler class"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        debaters: list[Debater],
        configuration: DebateConfig,
        summary_handler: SummaryHandler | None = None,
        metrics_handler: MetricsHandler | None = None,
        mediator: Mediator | None = None,
    ) -> None:
        """Initialize the debate instance.

        Args:
            model (LanguageModel): The language model to use.
            debaters (list[Debater]): The debaters participating in the debate.
            configuration (str, optional): The context of the debate.
            summary_handler (Summary | None, optional): The summary handler to use. Defaults to None.
            metrics_handler (MetricsHandler | None, optional): The metrics handler to use to compute message metrics. Defaults to None.
        """

        # Prompt context and metadata
        self.config = configuration

        # Positions
        self.prompt_for = "You are arguing in favor of the statement."
        self.prompt_against = "You are arguing against the statement."

        # Debater
        self.debaters = debaters
        self.mediator = mediator

        # Conversation detailed logs
        self.messages: list[Message] = []

        self.model = model
        self.metrics_handler = metrics_handler

        if summary_handler is None:
            self.summary_handler = SummaryHandler(model=model)
        else:
            self.summary_handler = summary_handler

    def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.
        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """
        for _ in track(range(rounds)):
            for index, debater in enumerate(self.debaters):

                if not self.should_participant_intervene(debater):
                    continue

                # Generate debater comment
                message = self.generate_debater_comment(debater)

                # Compute metrics if a handler is provided
                metrics = (
                    self.metrics_handler.compute_metrics(message)
                    if self.metrics_handler
                    else None
                )
                self.messages.append(Message(index, message, datetime.now(), metrics))
                self.summary_handler.update_with_message(message)

                if self.should_mediator_intervene():
                    # Generate mediator comment
                    message = self.generate_mediator_comment()

                    self.messages.append(
                        Message(None, message, datetime.now(), metrics)
                    )
                    # Include mediator messages in the summary
                    self.summary_handler.update_with_message(
                        f"Message from a mediator: {message}"
                    )

    ###############################################################################################
    #                                     HELPERS & SHORTHANDS                                    #
    ###############################################################################################

    def should_participant_intervene(self, debater: Debater) -> bool:
        """Shorthand helper to decide whether a participant should intervene in the debate."""

        return should_participant_intervene(
            model=self.model,
            config=self.config,
            summary=self.summary_handler,
            debater=debater,
        )

    def generate_debater_comment(self, debater: Debater) -> str:
        """Shorthand helper to generate a comment for the debater."""

        return debater_comment(
            model=self.model,
            config=self.config,
            debater=debater,
            summary=self.summary_handler,
        )

    def should_mediator_intervene(self) -> bool:
        """Decide whether the mediator should intervene in the debate."""

        if self.mediator is None:
            return False

        return should_mediator_intervene(
            model=self.model,
            config=self.config,
            mediator=self.mediator,
            summary=self.summary_handler,
        )

    def generate_mediator_comment(self) -> str:
        """Generate a comment for the mediator."""

        assert (
            self.mediator is not None
        ), "Trying to generate a mediator comment without mediator config"

        return mediator_comment(
            model=self.model,
            config=self.config,
            mediator=self.mediator,
            summary=self.summary_handler,
        )

    ###############################################################################################
    #                                        SERIALIZATION                                        #
    ###############################################################################################

    def pickle(self, path: str) -> None:
        """Serialize the debate configuration and logs to a pickle file.
        This does not include the model, summary handler, and other non data-relevant attributes.

        Args:
            path (str): The path to the pickle file.
        """

        data: "DebatePickle" = {
            "config": self.config,
            "debaters": self.debaters,
            "messages": self.messages,
        }

        with open(path, "wb") as file:
            pickle.dump(
                data,
                file,
            )

    @staticmethod
    def unpickle(path: str) -> "DebatePickle":
        """Load a debate configuration and logs from a pickle file.

        Args:
            path (str): The path to the pickle file.
        """

        with open(path, "rb") as file:
            return pickle.load(file)


class DebatePickle(TypedDict):
    """Pickled debate data"""

    config: DebateConfig
    debaters: list[Debater]
    messages: list[Message]
