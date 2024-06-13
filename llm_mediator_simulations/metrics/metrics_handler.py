"""Handler class to compute metrics for given input texts."""

from llm_mediator_simulations.metrics.criteria import (
    ArgumentQuality,
    measure_argument_quality,
)
from llm_mediator_simulations.metrics.perspective_api import PerspectiveScorer
from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.utils.types import Metrics


class MetricsHandler:
    """Handler class to compute metrics for given input texts."""

    def __init__(
        self,
        *,
        perspective: PerspectiveScorer | None,
        model: LanguageModel | None,
        argument_qualities: list[ArgumentQuality] | None
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
            metrics.argument_qualities = {}
            for argument_quality in self.argument_qualities:
                metrics.argument_qualities[argument_quality] = measure_argument_quality(
                    self.model, text, argument_quality
                )

        return metrics
