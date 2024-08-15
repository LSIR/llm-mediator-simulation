"""Mediator handler class."""

from datetime import datetime

from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.prompt import mediator_intervention
from llm_mediator_simulation.simulation.summary.handler import SummaryHandler
from llm_mediator_simulation.utils.probabilities import (
    ProbabilityMapper,
    ProbabilityMappingConfig,
)
from llm_mediator_simulation.utils.types import Intervention


class MediatorHandler:
    """Mediator simulation handler class"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        config: MediatorConfig,
        debate_config: DebateConfig,
        summary_handler: SummaryHandler,
        probability_config: ProbabilityMappingConfig | None = None,
    ) -> None:
        """Initialize the mediator handler.

        Args:
            model: The language model to use.
            config: The mediator configuration.
            debate_config: The debate configuration.
            summary_handler: The conversation summary handler.
            probability_config: The probability mapping config to use for monitoring mediator intervention. Defaults to None.
        """

        self.model = model
        self.config = config
        self.debate_config = debate_config
        self.summary_handler = summary_handler
        self.probability_mapper = (
            ProbabilityMapper(probability_config) if probability_config else None
        )

    def intervention(self) -> Intervention:
        """Do a mediator intervention."""

        response, prompt, do_intervene = mediator_intervention(
            model=self.model,
            config=self.debate_config,
            mediator=self.config,
            summary=self.summary_handler,
            probability_mapper=self.probability_mapper,
        )

        return Intervention(
            debater=None,
            text=response["text"] if do_intervene else None,
            prompt=prompt,
            justification=response["intervention_justification"],
            timestamp=datetime.now(),
        )
