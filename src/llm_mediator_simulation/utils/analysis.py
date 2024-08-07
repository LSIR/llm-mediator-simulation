"""Debate analysis utilities."""

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.simulation.configuration import (
    AxisPosition,
    PersonalityAxis,
)
from llm_mediator_simulation.simulation.debate import DebatePickle
from llm_mediator_simulation.utils.types import Intervention, Metrics


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

    personalities = []

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

    n = len(debate.debaters)

    aggregate: dict[PersonalityAxis, list[float]] = {}

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
            if axis not in aggregate:
                aggregate[axis] = [position.value]
            elif len(aggregate[axis]) <= round:
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


def aggregate_metrics(metrics: list[Metrics]):
    """Aggregate a list of metrics into plottable values."""

    # Aggregate perspective
    perspective = [m.perspective for m in metrics if m.perspective is not None]
    if len(perspective) == 0:
        perspective = None

    # Aggregate other metrics
    qualities: dict[ArgumentQuality, list[float]] = {}

    if len(metrics) == 0 or metrics[0].argument_qualities is None:
        return perspective, qualities

    for m in metrics:
        if m.argument_qualities is None:
            continue

        for quality, agreement in m.argument_qualities.items():
            if quality not in qualities:
                qualities[quality] = []
            qualities[quality].append(agreement.value)

    return perspective, qualities


def aggregate_average_metrics(debate: DebatePickle):
    """Aggregate a debate's metrics into an average among all debaters per round of interventions."""

    n = len(debate.debaters)

    round = 0
    debater_count = 0
    perspective: list[float] | None = []
    aggregate: dict[ArgumentQuality, list[float]] = {}

    for intervention in debate.interventions:

        if intervention.debater is None:
            continue

        debater_count += 1

        if intervention.metrics is None:
            continue

        if intervention.metrics.perspective is not None:
            if len(perspective) <= round:
                perspective.append(intervention.metrics.perspective)
            else:
                perspective[round] += intervention.metrics.perspective

        if intervention.metrics.argument_qualities is not None:
            for quality, agreement in intervention.metrics.argument_qualities.items():
                if quality not in aggregate:
                    aggregate[quality] = [agreement.value]
                elif len(aggregate[quality]) <= round:
                    aggregate[quality].append(agreement.value)
                else:
                    aggregate[quality][round] += agreement.value

        if debater_count == n:
            round += 1
            debater_count = 0

    # Compute the average for every axis
    perspective = [p / n for p in perspective]

    for quality, values in aggregate.items():
        aggregate[quality] = [v / n for v in values]

    if len(perspective) == 0:
        perspective = None

    return perspective, aggregate
