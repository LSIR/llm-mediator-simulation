"""An example analysis script run via CLI to analyze a pickled debate simulation."""

import click
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from llm_mediator_simulation.simulation.configuration import (
    AxisPosition,
    PersonalityAxis,
)
from llm_mediator_simulation.simulation.debate import (
    Debate,
    aggregate_personalities,
    personalities_of_name,
)


def common_options(func):
    """Common options for the analysis commands."""
    func = click.option(
        "--debate", "-d", help="The pickled debate to analyze.", required=True
    )(func)
    func = click.option(
        "--average", "-a", help="Average the values among debaters.", is_flag=True
    )(func)
    return func


@click.command("metrics")
@common_options
def metrics(debate: str, average: bool):
    """Plot the debater metrics"""

    data = Debate.unpickle(debate)

    print(debate, average)


@click.command("personalities")
@common_options
def personalities(debate: str, average: bool):
    """Plot the debater personalities"""

    data = Debate.unpickle(debate)
    n = len(data.debaters)

    if average:
        pass
        # TODO
    else:
        _, axs = plt.subplots(n, 1)
        for i in range(n):
            # Compute personalities
            debater_personalities = personalities_of_name(data, data.debaters[i].name)
            aggregate = aggregate_personalities(debater_personalities)

            ax: Axes = axs[i]  # type: ignore
            helper_plot(ax, aggregate, f"Personalities of {data.debaters[i].name}")

    plt.tight_layout()
    plt.show()


def helper_plot(
    axes: Axes, personalities: dict[PersonalityAxis, list[AxisPosition]], title: str
):
    """Helper function to plot personalities on a given axis."""

    axes.set_title(title)
    axes.set_xlabel("Interventions")
    axes.set_ylabel("Value (0 = left, 4 = right)")

    for axis, positions in personalities.items():
        values = [p.value for p in positions]
        axes.plot(
            range(len(positions)),
            values,
            label=f"{axis.value.name}: {axis.value.left} ↗ {axis.value.right}",
        )
        axes.legend()


@click.group()
def main():
    pass


main.add_command(metrics)
main.add_command(personalities)


if __name__ == "__main__":
    main()
