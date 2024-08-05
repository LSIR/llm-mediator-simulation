"""Online debate simulation handler class"""

import pickle
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

from rich.progress import track

from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.configuration import (
    AxisPosition,
    DebateConfig,
    Debater,
    Mediator,
    PersonalityAxis,
)
from llm_mediator_simulation.simulation.prompt import (
    debater_intervention,
    debater_personality_update,
    mediator_intervention,
)
from llm_mediator_simulation.simulation.summary_handler import SummaryHandler
from llm_mediator_simulation.utils.decorators import benchmark
from llm_mediator_simulation.utils.types import Intervention, LLMMessage


class Debate:
    """Online debate simulation handler class"""

    def __init__(
        self,
        *,
        debater_model: LanguageModel,
        mediator_model: LanguageModel,
        debaters: list[Debater],
        configuration: DebateConfig,
        summary_handler: SummaryHandler | None = None,
        metrics_handler: MetricsHandler | None = None,
        mediator: Mediator | None = None,
    ) -> None:
        """Initialize the debate instance.

        Args:
            debater_model: The language model to use for debaters.
            mediator_model: The language model to use for the mediator.
            debaters: The debaters participating in the debate.
            configuration: The context of the debate.
            summary_handler: The summary handler to use. Defaults to None.
            metrics_handler: The metrics handler to use to compute message metrics. Defaults to None.
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

        self.debater_model = debater_model
        self.mediator_model = mediator_model
        self.metrics_handler = metrics_handler

        if summary_handler is None:
            self.summary_handler = SummaryHandler(
                model=mediator_model, debaters=debaters
            )
        else:
            self.summary_handler = summary_handler

    def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.
        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """
        for _ in track(range(rounds)):
            for debater in self.debaters:

                intervention, prompt = self.debater_intervention(debater)

                if not intervention["do_intervene"]:
                    self.interventions.append(
                        Intervention(
                            deepcopy(debater),
                            None,
                            prompt,
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
                    deepcopy(debater),
                    message,
                    prompt,
                    intervention["intervention_justification"],
                    datetime.now(),
                    metrics,
                )

                self.interventions.append(intervention)
                self.summary_handler.add_new_message(intervention)

                if self.mediator is None:
                    self.summary_handler.regenerate_summary()
                    continue

                intervention, prompt = self.mediator_intervention()

                if intervention["do_intervene"]:
                    # Extract the mediator comment
                    message = intervention["text"]

                    intervention = Intervention(
                        None,
                        message,
                        prompt,
                        intervention["intervention_justification"],
                        datetime.now(),
                    )

                    self.interventions.append(intervention)
                    # Include mediator messages in the summary
                    self.summary_handler.add_new_message(intervention)
                    self.summary_handler.regenerate_summary()
                else:
                    self.interventions.append(
                        Intervention(
                            None,
                            None,
                            prompt,
                            intervention["intervention_justification"],
                            datetime.now(),
                        )
                    )

    ###############################################################################################
    #                                     HELPERS & SHORTHANDS                                    #
    ###############################################################################################

    @benchmark(name="Debater Intervention", verbose=False)
    def debater_intervention(self, debater: Debater) -> tuple[LLMMessage, str]:
        """Shorthand helper to decide whether a debater should intervene in the debate."""

        # Get the interventions of other debaters after this one
        last_interventions = self.interventions[1 - 2 * len(self.debaters) :]

        # Update the debater personality
        debater_personality_update(
            model=self.debater_model,
            debater=debater,
            interventions=last_interventions,
        )

        return debater_intervention(
            model=self.debater_model,
            config=self.config,
            summary=self.summary_handler,
            debater=debater,
        )

    @benchmark(name="Mediator Intervention", verbose=False)
    def mediator_intervention(self) -> tuple[LLMMessage, str]:
        """Shorthand helper to decide whether the mediator should intervene in the debate."""

        assert (
            self.mediator is not None
        ), "Trying to generate a mediator comment without mediator config"

        return mediator_intervention(
            model=self.mediator_model,
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
            path (str): The path to the pickle file, without file extension.
        """

        data: "DebatePickle" = DebatePickle(
            self.config, self.debaters, self.interventions
        )

        with open(f"{path}.pkl", "wb") as file:
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
    interventions: list[Intervention]


###################################################################################################
#                                           UTILITIES                                             #
###################################################################################################


def interventions_of_name(debate: DebatePickle, name: str) -> list[Intervention]:
    """Filter interventions from a debate, only keeping those from a specific debater."""
    return [
        intervention
        for intervention in debate.interventions
        if intervention.debater and intervention.debater.name == name
    ]


def personalities_of_name(
    debate: DebatePickle, name: str
) -> list[dict[PersonalityAxis, AxisPosition]]:
    """Extract a single debater's personalities from a debate."""

    debater = None
    for d in debate.debaters:
        if d.name == name:
            debater = d
            break

    if debater is None:
        raise ValueError(f"Debater {name} not found in the debate.")

    if not debater.personalities:
        raise ValueError(f"Debater {name} has no personalities.")

    personalities = [debater.personalities]

    for intervention in debate.interventions:
        if (
            intervention.debater
            and intervention.debater.name == name
            and intervention.debater.personalities
        ):
            personalities.append(intervention.debater.personalities)

    return personalities


def aggregate_personalities(personnalities: list[dict[PersonalityAxis, AxisPosition]]):
    """Aggregate a list of personalities into lists of personality positions for every personality
    (for plotting)"""

    aggregate: dict[PersonalityAxis, list[AxisPosition]] = {}

    for p in personnalities:
        for axis, position in p.items():
            if axis not in aggregate:
                aggregate[axis] = []
            aggregate[axis].append(position)

    return aggregate


def aggregate_average_personalities(debate: DebatePickle):
    """Aggregate the average of personalities for each round of interventions"""

    debaters = debate.debaters
    n = len(debaters)

    aggregate: dict[PersonalityAxis, list[float]] = {}

    # Compute initial average
    for debater in debaters:
        if debater.personalities is None:
            continue

        for axis, position in debater.personalities.items():
            if axis not in aggregate:
                aggregate[axis] = [position.value]
            else:
                aggregate[axis][0] += position.value

    # Compute average for each round
    round = 1
    debater_count = 0
    for intervention in debate.interventions:

        if intervention.debater is None:
            continue

        debater_count += 1

        if intervention.debater.personalities is None:
            continue

        for axis, position in intervention.debater.personalities.items():
            if len(aggregate[axis]) <= round:
                aggregate[axis].append(position.value)
            else:
                aggregate[axis][round] += position.value

        # Go to to next round
        if debater_count == n:
            round += 1
            debater_count = 0

    # Compute the average for every axis
    for axis, values in aggregate.items():
        aggregate[axis] = [v / n for v in values]

    return aggregate
