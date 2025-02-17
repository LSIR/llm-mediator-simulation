"""Plot graphs about the simulation results."""

import matplotlib.pyplot as plt

from llm_mediator_simulation.utils.types import Intervention


def plot_metrics(interventions: list[Intervention]) -> None:
    """Plot the metrics of a debate over time."""

    # 1. Agregate metrics
    metrics = {}

    for message in interventions:
        if message.text is None or message.text == "" or message.metrics is None:
            continue

        if message.metrics.perspective is not None:
            metrics.setdefault("perspective", []).append(message.metrics.perspective)

        if message.metrics.argument_qualities is not None:
            for quality, agreement in message.metrics.argument_qualities.items():
                metrics.setdefault(quality, []).append(agreement.value)

    if metrics["perspective"]:
        plt.plot(metrics["perspective"], label="Toxicity")
        plt.title("Toxicity over time")
        plt.xlabel("Message index")
        plt.ylabel("Toxicity level [0, 1]")
        plt.legend()
        plt.xticks(range(len(metrics["perspective"])))
        ax = plt.gca()
        ax.set_ylim([0, 1])  # type: ignore
        plt.show()

    for quality in metrics:
        if quality == "perspective":
            continue

        plt.plot(metrics[quality], label=quality)
    plt.title(f"Quality over time")
    plt.xlabel("Message index")
    plt.ylabel(f"Quality level")
    plt.yticks(range(1, 6))
    plt.xticks(range(len(metrics["perspective"])))
    plt.legend()
    plt.show()
