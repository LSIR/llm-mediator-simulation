"""An example analysis script run via CLI to analyze a pickled debate simulation.

The following commands are available:
(Note: if you replace the `debate` argument with a directory, the script will use the last debate in the directory.)

Plot the metrics of a debate:
```bash
python examples/example_analysis.py metrics -d debate.pkl
python examples/example_analysis.py metrics -d debate.pkl -a  # Averaged over debaters
```

Plot the personalities of debaters over time:
```bash
python examples/example_analysis.py personalities -d debate.pkl
python examples/example_analysis.py personalities -d debate.pkl -a  # Averaged over debaters
```

Generate a transcript of the debate:
```bash
python examples/example_analysis.py transcript -d debate.pkl
```

Print the debate data in a pretty format:
```bash
python examples/example_analysis.py print -d debate.pkl
```
"""

import os

import click
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from rich.console import Console
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


def get_last_debate_in_dir(dir: str) -> str:
    # Recursively search for the last debate in the directory
    # We assume that if the last recursive subdir does not contain any pickled files, then there is no debate
    # in the directory. We don't check previous subdir that may recursively contain pickled files...
    # If time, implement DFS traversal of the subdir tree...
    files = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and f.endswith(".pkl")
    ]

    while not files:
        # get the last subdirectory
        subdirs = [
            os.path.join(dir, d)
            for d in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, d))
        ]
        if not subdirs:
            raise ValueError(f"No pickled debates found in {dir}")

        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        dir = subdirs[0]
        files = [
            os.path.join(dir, f)
            for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f)) and f.endswith(".pkl")
        ]

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]  # Return the most recent file


def pickle_options(func):
    """Add a pickle option to the command."""

    # If debate is a directory, unpickle the last debate
    def wrapper(*args, **kwargs):
        if os.path.isdir(kwargs["debate"]):
            kwargs["debate"] = get_last_debate_in_dir(kwargs["debate"])
            print(f"Using the last debate in directory: {kwargs['debate']}")
        # elif os.path.isfile(args[0]):
        #     kwargs["debate"] = get_last_debate_in_dir(args[0])
        #     print(f"Using last debate in directory: {kwargs['debate']}")

        return func(*args, **kwargs)

    return click.option(
        "--debate", "-d", help="The pickled debate to analyze.", required=True
    )(wrapper)


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
    # assert that all debaters have personalities
    assert all(
        [debater.personality is not None for debater in data.debaters]
    ), "No personalities found for at least one debater"
    # Assert that debaters all have the same variable features
    assert (
        len(
            set(
                map(
                    frozenset,
                    [
                        debater.personality.variable_scale_set()
                        for debater in data.debaters
                        if debater.personality is not None
                    ],
                )
            )
        )
        == 1
    ), "Debaters have different variable features"

    if data.debaters[0].personality is not None:
        scale_variable = data.debaters[0].personality.number_of_scale_variables()
    else:
        raise ValueError("The personality of the first debater is None.")

    if average:
        _, axs = plt.subplots(scale_variable, 1, figsize=(10, 30))
        aggregated_personalities = aggregate_average_personalities(data)

        plot_personalities(
            axs, aggregated_personalities, "Average personality evolution", average=True
        )

    else:
        _, axs = plt.subplots(scale_variable, n, figsize=(10, 30))
        for j in range(n):
            # Compute personalities
            debater_personalities = personalities_of_name(data, data.debaters[j].name)
            aggregate = aggregate_personalities(debater_personalities)

            col_axes: Axes = axs[:, j]
            plot_personalities(
                col_axes,
                aggregate,
                f"Personality evolution of {data.debaters[j].name}",
                first_column=(j == 0),
            )
    plt.tight_layout()
    plt.show()
    debate_timestamp_str = debate.split("_")[-1].split(".")[0]
    if not debate_timestamp_str:
        raise ValueError("Expecting a debate name format like '*_YYYYMMDD-HHMMSS.pkl'")

    if not os.path.exists("plot"):
        os.makedirs("plot")
    if average:
        filename = f"plot/plot_personalities_average_{debate_timestamp_str}.png"
    else:
        filename = f"plot/plot_personalities_{debate_timestamp_str}.png"
    plt.savefig(filename)


@click.command("print")
@pickle_options
def pretty_print(debate: str):
    """Print the debate data in a pretty format."""
    if os.path.isdir(debate):
        debate = get_last_debate_in_dir(debate)

    data = DebateHandler.unpickle(debate)
    printable_data = data.to_printable()
    console = Console(force_terminal=True, record=True)
    pprint(printable_data, console=console)


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
