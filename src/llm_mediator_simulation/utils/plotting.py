"""Plotting utilities"""

from enum import Enum
from typing import Any, Mapping, Sequence, Type

import numpy
from matplotlib.axes import Axes

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.ideologies import (
    Ideology,
    Issues,
    MonoAxisIdeology,
)
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert5Level,
    Likert7AgreementLevel,
    Likert11LikelihoodLevel,
    Scale,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait


def plot_personalities(
    col_axes: Axes,
    personalities: Mapping[Any, Sequence[Scale]] | Mapping[Any, Sequence[float]],
    title: str,
    first_column: bool = True,
    average: bool = False,
):  # TODO For ideodologies, break continuity of plot for independent and libertarian in the case not aggregate.
    """Helper function to plot personalities on a given axis."""
    assert type(col_axes) is numpy.ndarray
    if average:
        assert type(next(iter(personalities.values()))[0]) is float
    else:
        assert type(next(iter(personalities.values()))[0]) is Scale
    col_axes[0].set_title(title)
    col_axes[-1].set_xlabel("Interventions")
    for i, (feature, values) in enumerate(personalities.items()):

        axes = col_axes[i]

        if average:
            if feature == "Ideology" or isinstance(feature, Issues):
                likert_scale = MonoAxisIdeology
            elif isinstance(feature, PersonalityTrait):
                likert_scale = Likert3Level
            elif isinstance(feature, PersonalityFacet):
                likert_scale = KeyingDirection
            elif isinstance(feature, MoralFoundation):
                likert_scale = Likert5Level
            elif isinstance(feature, BasicHumanValues):
                likert_scale = Likert5ImportanceLevel
            elif isinstance(feature, str):
                split_on_underscore = feature.split("_")
                assert split_on_underscore, "Feature name should be split on underscore"
                if split_on_underscore[0] == "agreement":
                    likert_scale = Likert7AgreementLevel
                elif split_on_underscore[0] == "belief":
                    likert_scale = Likert11LikelihoodLevel
                else:
                    raise ValueError(
                        f"Feature name {feature} does not match any known scale"
                    )
                feature = split_on_underscore[1:]

        else:
            likert_scale: Type[Enum] = type(values[0])
        likert_size = len(likert_scale)

        n = len(next(iter(personalities.values())))

        # axes.set_ylabel("Value")
        axes.set_ylim(-0.5, likert_size - 0.5)
        axes.set_xticks(range(n))
        axes.set_yticks(range(likert_size))
        if first_column:
            axes.set_yticklabels(
                [str(value.value.capitalize()) for value in likert_scale]
            )
        else:
            axes.set_yticklabels([])

        if average:
            numeric_values = values
        else:
            numeric_values = [list(likert_scale).index(value) for value in values]
        if isinstance(feature, str):
            label = feature
        else:
            label = feature.name.capitalize()
        axes.plot(range(len(values)), numeric_values, label=label)
        axes.legend()

        # Plot a middle line at the middle value
        if not average and likert_scale == Ideology:
            axes.axhline(y=(likert_size - 3) / 2, color="k", linestyle="--")
        else:
            axes.axhline(y=(likert_size - 1) / 2, color="k", linestyle="--")


def plot_metrics(
    axes: Axes,
    perspective: list[float] | None,
    qualities: dict[ArgumentQuality, list[float]],
    title: str,
):
    """Helper function to plot metrics on a given axis."""

    n_interventions = max(len(perspective or []), len(next(iter(qualities.values()))))

    axes.set_title(title)
    axes.set_xlabel("Interventions")
    axes.set_ylabel("Intensity")
    axes.set_ylim(-0.5, 4.5)
    axes.set_xticks(range(n_interventions))
    axes.set_yticks(range(5))

    if perspective is not None:
        # Scale to 0-4, like the argument qualities
        perspective = [p * 4 for p in perspective]
        axes.plot(range(len(perspective)), perspective, label="Perspective toxicity")
        axes.legend()

    for quality, agreements in qualities.items():
        axes.plot(range(len(agreements)), agreements, label=f"{quality.name}")
        axes.legend()

    # Plot a middle line at 2 (the middle value)
    axes.axhline(y=2, color="k", linestyle="--")
