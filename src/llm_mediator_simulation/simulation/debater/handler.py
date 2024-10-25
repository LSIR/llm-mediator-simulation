"""Debater handler class"""

from copy import deepcopy
from datetime import datetime

from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.simulation.prompt import (
    debater_intervention,
    debater_personality_update,
)
from llm_mediator_simulation.simulation.summary.handler import SummaryHandler
from llm_mediator_simulation.utils.types import Intervention


class DebaterHandler:
    """Debater simulation handler class"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        config: DebaterConfig,
        debate_config: DebateConfig,
        summary_handler: SummaryHandler,
    ) -> None:
        """Initialize the debater handler.

        Args:
            model: The language model to use.
            config: The debater configuration. The debater personality will evolve during the debate.
            debate_config: The debate configuration.
            summary_handler: The conversation summary handler.
        """
        self.model = model
        self.config = config
        self.debate_config = debate_config
        self.summary_handler = summary_handler

    def intervention(self, 
                     update_personality=False, 
                     seed: int | None = None) -> Intervention:
        """Do a debater intervention.

        Args:
            update_personality: Whether to update the debater personality based on the last messages before intervention.
        """

        # Update the debater personality
        if update_personality:
            debater_personality_update(
                model=self.model,
                debater=self.config,
                interventions=self.summary_handler.latest_messages,
            )

        response, prompt = debater_intervention(
            model=self.model,
            config=self.debate_config,
            summary=self.summary_handler,
            debater=self.config,
            seed=seed,
        )

        return Intervention(
            debater=deepcopy(self.config),  # Freeze the debater configuration
            text=response["text"],
            prompt=prompt,
            justification=response["intervention_justification"],
            timestamp=datetime.now(),
        )

    def snapshot_personality(self) -> DebaterConfig:
        """Snapshot the current debater personality."""
        return deepcopy(self.config)
