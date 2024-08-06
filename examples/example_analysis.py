"""An example analysis script run via CLI to analyze a pickled debate simulation."""

import click
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from llm_mediator_simulation.simulation.debate import Debate
from llm_mediator_simulation.utils.analysis import (
    aggregate_average_personalities,
    aggregate_personalities,
    interventions_of_name,
    personalities_of_name,
)
from llm_mediator_simulation.utils.plotting import plot_metrics, plot_personalities
from llm_mediator_simulation.utils.types import Metrics


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
    n = len(data.debaters)

    if average:
        pass  # TODO
    else:
        _, axs = plt.subplots(n, 1)
        for i, debater in enumerate(data.debaters):
            interventions = interventions_of_name(data, debater.name)
            axes: Axes = axs[i]  # type: ignore

            metrics = [
                intervention.metrics
                for intervention in interventions
                if intervention.metrics is not None
            ]

            plot_metrics(axes, metrics, f"Metrics of {debater.name}")

    plt.tight_layout()
    plt.show()

    print(debate, average)


@click.command("personalities")
@common_options
def personalities(debate: str, average: bool):
    """Plot the debater personalities"""

    data = Debate.unpickle(debate)
    n = len(data.debaters)

    if average:
        aggregate = aggregate_average_personalities(data)
        axes = plt.gca()
        plot_personalities(axes, aggregate, "Average personalities")

    else:
        _, axs = plt.subplots(n, 1)
        for i in range(n):
            # Compute personalities
            debater_personalities = personalities_of_name(data, data.debaters[i].name)
            aggregate = aggregate_personalities(debater_personalities)

            axes: Axes = axs[i]  # type: ignore
            plot_personalities(
                axes, aggregate, f"Personalities of {data.debaters[i].name}"
            )
    plt.tight_layout()
    plt.show()


@click.group()
def main():
    pass


main.add_command(metrics)
main.add_command(personalities)


if __name__ == "__main__":
    main()
