"""Online debate simulation handler class"""

import pickle
from datetime import datetime
from typing import TypedDict

from rich.progress import track

from llm_mediator_simulations.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.configuration import DebateConfig, Debater
from llm_mediator_simulations.simulation.prompt import (
    debater_comment,
    should_participant_intervene,
)
from llm_mediator_simulations.simulation.summary import Summary
from llm_mediator_simulations.utils.types import Message


class Debate:
    """Online debate simulation handler class"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        debaters: list[Debater],
        configuration: DebateConfig,
        summary_handler: Summary | None = None,
        metrics_handler: MetricsHandler | None = None,
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

        # Conversation detailed logs
        self.messages: list[Message] = []

        self.model = model
        self.metrics_handler = metrics_handler

        if summary_handler is None:
            self.summary_handler = Summary(model=model)
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

                # TODO: here, decide if a mediator wants to intervene
                # define the decision + prompt in prompt.py, and see if it works well

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

    def unpickle(self, path: str) -> "DebatePickle":
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
