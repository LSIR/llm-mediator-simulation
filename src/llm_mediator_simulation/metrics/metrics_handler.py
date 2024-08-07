"""Handler class to compute metrics for given input texts."""

from concurrent.futures import ProcessPoolExecutor

from llm_mediator_simulation.metrics.criteria import (
    ArgumentQuality,
    async_measure_argument_qualities,
    measure_argument_qualities,
)
from llm_mediator_simulation.metrics.perspective_api import PerspectiveScorer
from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from llm_mediator_simulation.utils.types import Metrics


class MetricsHandler:
    """Handler class to compute metrics for given input texts."""

    def __init__(
        self,
        *,
        perspective: PerspectiveScorer | None = None,
        model: LanguageModel | None = None,
        argument_qualities: list[ArgumentQuality] | None = None,
    ) -> None:
        """Initialize the metrics handler instance.

        Args:
            perspective (PerspectiveScorer | None, optional): The Perspective API scorer. Computes the Perspective API toxicity score. Defaults to None.
            model (LanguageModel | None, optional): The language model to use for custom LLM-based metrics. Requires `argument_qualitites to be set`. Defaults to None.
            argument_qualities (list[ArgumentQuality] | None, optional): The argument qualities to evaluate. Requires `model` to be set. Defaults to None.
        """

        assert (model is None and argument_qualities is None) or (
            model is not None and argument_qualities is not None
        ), "Model and argument_qualities must be set together."

        self.perspective = perspective
        self.model = model
        self.argument_qualities = argument_qualities

    def compute_metrics(self, text: str) -> Metrics:
        """Compute the metrics for the given text.

        Args:
            text (str): The text to compute the metrics for.

        Returns:
            Metrics: The computed metrics.
        """

        metrics = Metrics()

        # Measure Perspective API toxicity
        if self.perspective is not None:
            metrics.perspective = self.perspective.score(text)

        # Measure custom LLM-based metrics
        if self.model is not None and self.argument_qualities is not None:
            metrics.argument_qualities = measure_argument_qualities(
                self.model, text, self.argument_qualities
            )

        return metrics


class AsyncMetricsHandler:
    """Handler class to compute metrics for given input texts asynchronously."""

    def __init__(
        self,
        *,
        perspective: PerspectiveScorer | None = None,
        model: AsyncLanguageModel | None = None,
        argument_qualities: list[ArgumentQuality] | None = None,
    ) -> None:
        """Initialize the metrics handler instance.

        Args:
            perspective (PerspectiveScorer | None, optional): The Perspective API scorer. Computes the Perspective API toxicity score. Defaults to None.
            model (LanguageModel | None, optional): The language model to use for custom LLM-based metrics. Requires `argument_qualitites to be set`. Defaults to None.
            argument_qualities (list[ArgumentQuality] | None, optional): The argument qualities to evaluate. Requires `model` to be set. Defaults to None.
        """

        assert (model is None and argument_qualities is None) or (
            model is not None and argument_qualities is not None
        ), "Model and argument_qualities must be set together."

        self.perspective = perspective
        self.model = model
        self.argument_qualities = argument_qualities

    async def compute_metrics(self, texts: list[str]) -> list[Metrics]:

        metrics: list[Metrics] = [Metrics() for _ in range(len(texts))]

        # Measure Perspective API toxicity
        if self.perspective is not None:
            # Perspective API does not have an async client. We have to use multiprocessing to mimic it...
            with ProcessPoolExecutor() as executor:
                scores = list(executor.map(self.perspective.score, texts))

            for index in range(len(metrics)):
                metrics[index].perspective = scores[index]

        # Measure custom LLM-based metrics
        if self.model is not None and self.argument_qualities is not None:

            qualities = await async_measure_argument_qualities(
                self.model, texts, self.argument_qualities
            )
            for index in range(len(metrics)):
                metrics[index].argument_qualities = qualities[index]

        return metrics
