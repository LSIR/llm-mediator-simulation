"""Handler class to compute metrics for given input texts."""

from llm_mediator_simulation.metrics.criteria import (
    ArgumentQuality,
    measure_argument_qualities,
)
from llm_mediator_simulation.metrics.perspective_api import PerspectiveScorer
from llm_mediator_simulation.metrics.distinct_repetition_bertScore import Distinct3, Repetition4, LexicalRepetition, BERTScore
from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.utils.types import Intervention, Metrics


class MetricsHandler:
    """Handler class to compute metrics for given input texts."""

    def __init__(
        self,
        *,
        perspective: PerspectiveScorer | None = None,
        distinct3: Distinct3 | None = None,
        repetition4 : Repetition4 | None = None,
        lexicalrep : LexicalRepetition | None = None,
        bertscore : BERTScore | None = None,
        model: LanguageModel | None = None,
        argument_qualities: list[ArgumentQuality] | None = None,
    ) -> None:
        """Initialize the metrics handler instance.

        Args:
            perspective (PerspectiveScorer | None, optional): The Perspective API scorer. Computes the Perspective API toxicity score. Defaults to None.
            distinct3 (Distinct3 | None, optional): The Distinct-3 scorer. Computes the Distinct-3 metric. Defaults to None.
            repetition4 (Repetition4 | None, optional): The Repetition-4 scorer. Computes the Repetition-4 metric. Defaults to None.
            lexicalrep (LexicalRepetition | None, optional): The Lexical Repetition scorer. Computes the Lexical Repetition metric. Defaults to None.
            bertscore (BERTScore | None, optional): The BERTScore scorer. Computes the BERTScore metric. Defaults to None.
            model (LanguageModel | None, optional): The language model to use for custom LLM-based metrics. Requires `argument_qualities` to be set. Defaults to None.
            argument_qualities (list[ArgumentQuality] | None, optional): The argument qualities to evaluate. Requires `model` to be set. Defaults to None.
        """

        assert (model is None and argument_qualities is None) or (
            model is not None and argument_qualities is not None
        ), "Model and argument_qualities must be set together."

        self.perspective = perspective
        self.distinct3 = distinct3
        self.repetition4 = repetition4
        self.lexicalrep = lexicalrep
        self.bertscore = bertscore
        self.model = model
        self.argument_qualities = argument_qualities
        
        
    

    def compute_metrics(self, text: str, latest_messages: str) -> Metrics:
        """Compute the metrics for the given text."""
        """
        Args:
            text (str): The text to compute the metrics for.

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

            
        if self.distinct3 is not None:
            metrics.distinct3 = self.distinct3.score(latest_messages)
            
        if self.repetition4 is not None:
            metrics.repetition4 = self.repetition4.score(latest_messages)
        
        if self.lexicalrep is not None:
            metrics.lexicalrep = self.lexicalrep.score(latest_messages)
            
        if self.bertscore is not None:
            metrics.bertscore = self.bertscore.score(text)
            
        return metrics

        
        

    def inject_metrics(self, intervention: Intervention, latest_messages: str) -> None:
        """Inject metrics in place into a given intervention."""
        if intervention.text is not None:
            intervention.metrics = self.compute_metrics(text= intervention.text, latest_messages= latest_messages)
 
