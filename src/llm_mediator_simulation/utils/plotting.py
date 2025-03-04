"""Plotting utilities"""

from enum import Enum
from typing import Any, Mapping, Sequence, Type

import numpy
from matplotlib.axes import Axes

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.personalities.ideologies import Ideology
from llm_mediator_simulation.personalities.scales import Scale


def plot_personalities(
    col_axes: Axes,
    personalities: Mapping[Any, Sequence[Scale]],
    title: str,
    first_column: bool = True,
):
    """Helper function to plot personalities on a given axis."""
    assert type(col_axes) is numpy.ndarray
    col_axes[0].set_title(title)
    col_axes[-1].set_xlabel("Interventions")
    for i, (feature, values) in enumerate(personalities.items()):

        axes = col_axes[i]
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

        numeric_values = [list(likert_scale).index(value) for value in values]
        if isinstance(feature, str):
            label = feature
        else:
            label = feature.name.capitalize()
        axes.plot(range(len(values)), numeric_values, label=label)
        axes.legend()

        # Plot a middle line at the middle value
        if likert_scale == Ideology:
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
