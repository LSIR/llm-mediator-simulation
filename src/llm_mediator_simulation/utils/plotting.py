"""Plotting utilities"""

from typing import Mapping, Sequence

from matplotlib.axes import Axes

from llm_mediator_simulation.simulation.configuration import (
    AxisPosition,
    PersonalityAxis,
)
from llm_mediator_simulation.utils.analysis import aggregate_metrics
from llm_mediator_simulation.utils.types import Metrics


def plot_personalities(
    axes: Axes,
    personalities: Mapping[PersonalityAxis, Sequence[AxisPosition | float]],
    title: str,
):
    """Helper function to plot personalities on a given axis."""

    n = len(next(iter(personalities.values())))

    axes.set_title(title)
    axes.set_xlabel("Interventions")
    axes.set_ylabel("Value (0 = left, 4 = right)")
    axes.set_ylim(-0.5, 4.5)
    axes.set_xticks(range(n))
    axes.set_yticks(range(5))

    for axis, positions in personalities.items():
        values = [p if isinstance(p, (int, float)) else p.value for p in positions]
        axes.plot(
            range(len(positions)),
            values,
            label=f"{axis.value.name}: {axis.value.left} ↗ {axis.value.right}",
        )
        axes.legend()

    # Plot a middle line at 2 (the middle value)
    axes.axhline(y=2, color="k", linestyle="--")


def plot_metrics(axes: Axes, metrics: list[Metrics], title: str):
    """Helper function to plot metrics on a given axis."""

    perspective, qualities = aggregate_metrics(metrics)

    axes.set_title(title)
    axes.set_xlabel("Interventions")
    axes.set_ylabel("Intensity")
    axes.set_ylim(-0.5, 4.5)
    axes.set_xticks(range(len(metrics)))
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
