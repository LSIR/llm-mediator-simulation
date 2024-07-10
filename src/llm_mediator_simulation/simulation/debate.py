"""Online debate simulation handler class"""

import pickle
from dataclasses import dataclass
from datetime import datetime

from rich.progress import track

from llm_mediator_simulations.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.configuration import (
    DebateConfig,
    Debater,
    Mediator,
)
from llm_mediator_simulations.simulation.prompt import (
    debater_intervention,
    mediator_intervention,
)
from llm_mediator_simulations.simulation.summary_handler import SummaryHandler
from llm_mediator_simulations.utils.decorators import benchmark
from llm_mediator_simulations.utils.types import Intervention, LLMMessage


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
        self.interventions: list[Intervention] = []

        self.model = model
        self.metrics_handler = metrics_handler

        if summary_handler is None:
            self.summary_handler = SummaryHandler(model=model, debaters=debaters)
        else:
            self.summary_handler = summary_handler

    def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.
        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """
        for _ in track(range(rounds)):
            for index, debater in enumerate(self.debaters):

                intervention = self.debater_intervention(debater)

                if not intervention["do_intervene"]:
                    self.interventions.append(
                        Intervention(
                            index,
                            None,
                            intervention["intervention_justification"],
                            datetime.now(),
                        )
                    )
                    continue

                # Extract the debater comment
                message = intervention["text"]

                # Compute metrics if a handler is provided
                metrics = (
                    self.metrics_handler.compute_metrics(message)
                    if self.metrics_handler
                    else None
                )
                intervention = Intervention(
                    index,
                    message,
                    intervention["intervention_justification"],
                    datetime.now(),
                    metrics,
                )

                self.interventions.append(intervention)
                self.summary_handler.update_with_message(intervention)

                intervention = self.mediator_intervention()

                if intervention["do_intervene"]:
                    # Extract the mediator comment
                    message = intervention["text"]

                    intervention = Intervention(
                        None,
                        message,
                        intervention["intervention_justification"],
                        datetime.now(),
                        metrics,
                    )

                    self.interventions.append(intervention)
                    # Include mediator messages in the summary
                    self.summary_handler.update_with_message(intervention)
                else:
                    self.interventions.append(
                        Intervention(
                            None,
                            None,
                            intervention["intervention_justification"],
                            datetime.now(),
                        )
                    )

    ###############################################################################################
    #                                     HELPERS & SHORTHANDS                                    #
    ###############################################################################################

    @benchmark(name="Debater Intervention", verbose=False)
    def debater_intervention(self, debater: Debater) -> LLMMessage:
        """Shorthand helper to decide whether a debater should intervene in the debate."""

        return debater_intervention(
            model=self.model,
            config=self.config,
            summary=self.summary_handler,
            debater=debater,
        )

    @benchmark(name="Mediator Intervention", verbose=False)
    def mediator_intervention(self) -> LLMMessage:
        """Shorthand helper to decide whether the mediator should intervene in the debate."""

        assert (
            self.mediator is not None
        ), "Trying to generate a mediator comment without mediator config"

        return mediator_intervention(
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

        data: "DebatePickle" = DebatePickle(
            self.config, self.debaters, self.interventions
        )

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


@dataclass
class DebatePickle:
    """Pickled debate data"""

    config: DebateConfig
    debaters: list[Debater]
    messages: list[Intervention]
