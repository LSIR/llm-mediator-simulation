"""Debate analysis utilities."""

from enum import Enum
from typing import Any, Mapping, Type

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.personalities.ideologies import Ideology
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert5Level,
    Scale,
)
from llm_mediator_simulation.simulation.debate.handler import DebatePickle
from llm_mediator_simulation.utils.types import Intervention, Metrics

PERSONALITY_FIELD_NAMES = [
    "traits",
    "facets",
    "moral_foundations",
    "basic_human_values",
    "ideologies",
    "agreement_with_statements",
    "likelihood_of_beliefs",
]


def interventions_of_name(debate: DebatePickle, name: str) -> list[Intervention]:
    """Filter interventions from a debate, only keeping those from a specific debater."""
    return [
        intervention
        for intervention in debate.interventions
        if intervention.debater and intervention.debater.name == name
    ]


def personalities_of_name(debate: DebatePickle, name: str) -> list[Personality]:
    """Extract a single debater's personalities from a debate."""

    personalities = []

    for intervention in debate.interventions:
        if (
            intervention.debater
            and intervention.debater.name == name
            and intervention.debater.personality is not None
        ):
            personalities.append(intervention.debater.personality)

    return personalities


def aggregate_personalities(personnalities: list[Personality]):
    """Aggregate a list of personalities into lists of personality features for every personality feature
    (for plotting)"""

    aggregate: Mapping[Any, list[Scale]] = {}

    for debater_personality in personnalities:
        for personality_field_name in PERSONALITY_FIELD_NAMES:  # e.g. "traits"
            if getattr(debater_personality, f"variable_{personality_field_name}"):
                personality_field = getattr(
                    debater_personality, personality_field_name
                )  # e.g. debater_personality.traits
                if isinstance(personality_field, Ideology):
                    if "Ideology" not in aggregate:
                        aggregate["Ideology"] = []
                    aggregate["Ideology"].append(personality_field)

                elif isinstance(personality_field, list):
                    for feature in personality_field:
                        if personality_field_name == "traits":
                            default_value = Likert3Level.HIGH
                        elif personality_field_name == "facets":
                            default_value = KeyingDirection.POSITIVE
                        elif personality_field_name == "moral_foundations":
                            default_value = Likert5Level.FAIRLY
                        elif personality_field_name == "basic_human_values":
                            default_value = Likert5ImportanceLevel.IMPORTANT
                        else:
                            raise ValueError(
                                f"Personality field {personality_field_name} cannot be a list."
                            )
                        if feature not in aggregate:
                            aggregate[feature] = []
                        aggregate[feature].append(default_value)

                elif isinstance(personality_field, dict):
                    for feature, value in personality_field.items():
                        if feature not in aggregate:
                            aggregate[feature] = []
                        aggregate[feature].append(value)

                else:
                    raise ValueError(
                        f"Personality field {personality_field_name} is neither a list nor a dict."
                    )

    return aggregate


def aggregate_average_personalities(debate: DebatePickle):
    """Aggregate the average of personality feature for each round of interventions"""

    n = len(debate.debaters)

    aggregate: Mapping[Any, list[float]] = {}

    # Compute average for each round
    round = 0
    debater_count = 0
    for intervention in debate.interventions:
        if intervention.debater is None:
            continue

        debater_count += 1

        assert (
            intervention.debater.personality is not None
        ), "Debater has no personality"

        debater_personality = intervention.debater.personality

        for personality_field_name in PERSONALITY_FIELD_NAMES:  # e.g. "traits"
            if getattr(debater_personality, f"variable_{personality_field_name}"):
                personality_field: Ideology | list[Any] | Mapping[Any, Scale] = getattr(
                    debater_personality, personality_field_name
                )  # e.g. debater_personality.traits
                if isinstance(personality_field, Ideology):
                    if "Ideology" not in aggregate:
                        aggregate["Ideology"] = []
                    if personality_field == Ideology.EXTREMELY_LIBERAL:
                        numeric_value = 0
                    elif personality_field == Ideology.LIBERAL:
                        numeric_value = 1
                    elif personality_field == Ideology.SLIGHTLY_LIBERAL:
                        numeric_value = 2
                    elif personality_field in [
                        Ideology.MODERATE,
                        Ideology.INDEPENDENT,
                        Ideology.LIBERTARIAN,
                    ]:
                        numeric_value = 3
                    elif personality_field == Ideology.SLIGHTLY_CONSERVATIVE:
                        numeric_value = 4
                    elif personality_field == Ideology.CONSERVATIVE:
                        numeric_value = 5
                    elif personality_field == Ideology.EXTREMELY_CONSERVATIVE:
                        numeric_value = 6
                    else:
                        raise ValueError(
                            f"Personality field {personality_field_name} cannot be a list."
                        )
                    if len(aggregate["Ideology"]) <= round:
                        aggregate["Ideology"].append(numeric_value)
                    else:
                        aggregate["Ideology"][round] += numeric_value

                elif isinstance(personality_field, list):
                    for feature in personality_field:
                        if personality_field_name == "traits":
                            default_value = 2  # Likert3Level.HIGH
                        elif personality_field_name == "facets":
                            default_value = 1  # KeyingDirection.POSITIVE
                        elif personality_field_name == "moral_foundations":
                            default_value = 3  # Likert5Level.FAIRLY
                        elif personality_field_name == "basic_human_values":
                            default_value = 3  # Likert5ImportanceLevel.IMPORTANT
                        else:
                            raise ValueError(
                                f"Personality field {personality_field_name} cannot be a list."
                            )
                        if feature not in aggregate:
                            aggregate[feature] = []
                        if len(aggregate[feature]) <= round:
                            aggregate[feature].append(default_value)
                        else:
                            aggregate[feature][round] += default_value

                elif isinstance(personality_field, dict):
                    for feature, value in personality_field.items():
                        if personality_field_name == "agreement_with_statements":
                            # prepend feature with "agreement_"
                            feature = f"agreement_{feature}"
                        elif personality_field_name == "likelihood_of_beliefs":
                            # prepend feature with "belief_"
                            feature = f"belief_{feature}"

                        if feature not in aggregate:
                            aggregate[feature] = []

                        likert_scale: Type[Enum] = type(value)
                        numeric_value = list(likert_scale).index(value)

                        if len(aggregate[feature]) <= round:
                            aggregate[feature].append(numeric_value)
                        else:
                            aggregate[feature][round] += numeric_value

                else:
                    raise ValueError(
                        f"Personality field {personality_field_name} is neither a list nor a dict."
                    )

        # Go to to next round
        if debater_count == n:
            round += 1
            debater_count = 0
    # Compute the average for every axis
    for feature, values in aggregate.items():
        aggregate[feature] = [v / n for v in values]

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
