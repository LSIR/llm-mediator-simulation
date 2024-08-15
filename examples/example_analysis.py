"""An example analysis script run via CLI to analyze a pickled debate simulation.

The following commands are available:

Plot the metrics of a debate:
```bash
python examples/example_analysis.py metrics -d debate.pickle
python examples/example_analysis.py metrics -d debate.pickle -a  # Averaged over debaters
```

Plot the personalities of debaters over time:
```bash
python examples/example_analysis.py personalities -d debate.pickle
python examples/example_analysis.py personalities -d debate.pickle -a  # Averaged over debaters
```

Generate a transcript of the debate:
```bash
python examples/example_analysis.py transcript -d debate.pickle
```

Print the debate data in a pretty format:
```bash
python examples/example_analysis.py print -d debate.pickle
```
"""

import click
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from rich.pretty import pprint

from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.utils.analysis import (
    aggregate_average_metrics,
    aggregate_average_personalities,
    aggregate_metrics,
    aggregate_personalities,
    interventions_of_name,
    personalities_of_name,
)
from llm_mediator_simulation.utils.plotting import plot_metrics, plot_personalities
from llm_mediator_simulation.visualization.transcript import debate_transcript


def pickle_options(func):
    """Add a pickle option to the command."""
    return click.option(
        "--debate", "-d", help="The pickled debate to analyze.", required=True
    )(func)


def common_options(func):
    """Common options for the analysis commands."""
    func = pickle_options(func)
    func = click.option(
        "--average", "-a", help="Average the values among debaters.", is_flag=True
    )(func)
    return func


@click.command("metrics")
@common_options
def metrics(debate: str, average: bool):
    """Plot the debater metrics"""

    data = DebateHandler.unpickle(debate)
    n = len(data.debaters)

    if average:
        perspective, qualities = aggregate_average_metrics(data)

        axes = plt.gca()

        plot_metrics(axes, perspective, qualities, "Average metrics")

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

            perspective, qualities = aggregate_metrics(metrics)
            plot_metrics(axes, perspective, qualities, f"Metrics of {debater.name}")

    plt.tight_layout()
    plt.show()


@click.command("personalities")
@common_options
def personalities(debate: str, average: bool):
    """Plot the debater personalities"""

    data = DebateHandler.unpickle(debate)
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


@click.command("print")
@pickle_options
def pretty_print(debate: str):
    """Print the debate data in a pretty format."""

    data = DebateHandler.unpickle(debate)
    pprint(data)


@click.command("transcript")
@pickle_options
def transcript(debate: str):
    """Print the debate transcript.
    You can pipe it to a file to save it."""
    data = DebateHandler.unpickle(debate)

    print(debate_transcript(data))


@click.group()
def main():
    pass


main.add_command(metrics)
main.add_command(personalities)
main.add_command(pretty_print)
main.add_command(transcript)


if __name__ == "__main__":
    main()
