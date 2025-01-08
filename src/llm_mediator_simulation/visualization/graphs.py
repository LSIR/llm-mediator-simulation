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
            
        if message.metrics.distinct3 is not None:
            metrics.setdefault("distinct3", []).append(message.metrics.distinct3)

        if message.metrics.repetition4 is not None:
            metrics.setdefault("repetition4", []).append(message.metrics.repetition4)

        if message.metrics.lexicalrep is not None:
            metrics.setdefault("lexicalrep", []).append(message.metrics.lexicalrep)

        if message.metrics.bertscore is not None:
            metrics.setdefault("bertscore", []).append(message.metrics.bertscore)

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
        
        
    if metrics["distinct3"]:
        plt.plot(metrics["distinct3"], label="Distinct-3")
        plt.title("Distinct-3 over time")
        plt.xlabel("Message index")
        plt.ylabel("Distinct-3 level")
        plt.legend()
        plt.xticks(range(len(metrics["distinct3"])))
        plt.show()

    if metrics["repetition4"]:
        plt.plot(metrics["repetition4"], label="Repetition-4")
        plt.title("Repetition-4 over time")
        plt.xlabel("Message index")
        plt.ylabel("Repetition-4 level")
        plt.legend()
        plt.xticks(range(len(metrics["repetition4"])))
        plt.show()

    if metrics["lexicalrep"]:
        plt.plot(metrics["lexicalrep"], label="Lexical Repetition")
        plt.title("Lexical Repetition over time")
        plt.xlabel("Message index")
        plt.ylabel("Lexical Repetition level")
        plt.legend()
        plt.xticks(range(len(metrics["lexicalrep"])))
        plt.show()

    if metrics["bertscore"]:
        plt.plot(metrics["bertscore"], label="BERTScore")
        plt.title("BERTScore over time")
        plt.xlabel("Message index")
        plt.ylabel("BERTScore level")
        plt.legend()
        plt.xticks(range(len(metrics["bertscore"])))
        plt.show()

    for quality in metrics:
        if quality in ["perspective", "distinct3", "repetition4", "lexicalrep", "bertscore"]:
            continue

        plt.plot(metrics[quality], label=quality)
    plt.title(f"Quality over time")
    plt.xlabel("Message index")
    plt.ylabel(f"Quality level")
    plt.yticks(range(1, 6))
    plt.xticks(range(len(metrics["perspective"])))
    plt.legend()
    plt.show()
