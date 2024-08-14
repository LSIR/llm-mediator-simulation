"""Async handler class to compute metrics for given input texts."""

from concurrent.futures import ProcessPoolExecutor

from llm_mediator_simulation.metrics.criteria import (
    ArgumentQuality,
    async_measure_argument_qualities,
)
from llm_mediator_simulation.metrics.perspective_api import PerspectiveScorer
from llm_mediator_simulation.models.language_model import AsyncLanguageModel
from llm_mediator_simulation.utils.types import Intervention, Metrics


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

    async def inject_metrics(self, interventions: list[Intervention]) -> None:
        """Inject metrics in place into a given list of interventions."""

        # Do not compute metrics for empty interventions
        valid_indexes = [
            i for i, intervention in enumerate(interventions) if intervention.text
        ]
        texts: list[str] = [interventions[i].text for i in valid_indexes]  # type: ignore

        metrics = await self.compute_metrics(texts)

        for index, metric in zip(valid_indexes, metrics):
            interventions[index].metrics = metric
