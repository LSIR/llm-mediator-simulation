"""Asynchronous mediator handler class."""

from datetime import datetime

from llm_mediator_simulation.models.language_model import AsyncLanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.prompt import async_mediator_interventions
from llm_mediator_simulation.simulation.summary.async_handler import AsyncSummaryHandler
from llm_mediator_simulation.utils.probabilities import (
    ProbabilityMapper,
    ProbabilityMappingConfig,
)
from llm_mediator_simulation.utils.types import Intervention


class AsyncMediatorHandler:
    """Mediator simulation handler class"""

    def __init__(
        self,
        *,
        model: AsyncLanguageModel,
        config: MediatorConfig,
        debate_config: DebateConfig,
        summary_handler: AsyncSummaryHandler,
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

    async def interventions(
        self, valid_indexes: list[int], seed: int | None = None
    ) -> list[Intervention]:
        """Do a mediator intervention on every debate.

        Args:
            valid_indexes: The debate indexes which must have a mediator intervention.
        """

        results, prompts = await async_mediator_interventions(
            model=self.model,
            config=self.debate_config,
            mediator=self.config,
            summary=self.summary_handler,
            valid_indexes=valid_indexes,
            seed=seed,
        )

        return [
            Intervention(
                debater=None,
                text=result["text"],
                prompt=prompt,
                justification=result["intervention_justification"],  # TODO Update
                timestamp=datetime.now(),
            )
            for result, prompt in zip(results, prompts)
        ]
