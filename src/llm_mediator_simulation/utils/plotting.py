"""Plotting utilities"""

from typing import Mapping, Sequence

from matplotlib.axes import Axes

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.simulation.debater.config import (
    AxisPosition,
    PersonalityAxis,
)


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

def plot_metrics(
    axes: Axes,
    perspective: list[float] | None,
    distinct3: list[float] | None,
    repetition4: list[float] | None,
    lexicalrep: list[float] | None,
    bertscore: list[float] | None,
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
        
    if distinct3 is not None:
        axes.plot(range(len(distinct3)), distinct3, label="Distinct-3")
        
    if repetition4 is not None:
        axes.plot(range(len(repetition4)), repetition4, label="Repetition-4")
        
    if lexicalrep is not None:
        axes.plot(range(len(lexicalrep)), lexicalrep, label="Lexical Repetition")
        
    if bertscore is not None:
        axes.plot(range(len(bertscore)), bertscore, label="BERTScore")
        
    for quality, agreements in qualities.items():
        axes.plot(range(len(agreements)), agreements, label=f"{quality.name}")
        
    # Plot a middle line at 2 (the middle value)
    axes.axhline(y=2, color="k", linestyle="--")

    # Adjust legend font size
    axes.legend(fontsize='x-small')