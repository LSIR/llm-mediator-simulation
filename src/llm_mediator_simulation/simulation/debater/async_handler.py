"""Asynchronous debater handler class"""

from copy import deepcopy
from datetime import datetime

from llm_mediator_simulation.models.language_model import AsyncLanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.simulation.prompt import (
    async_debater_interventions,
    async_debater_personality_update,
)
from llm_mediator_simulation.simulation.summary.async_handler import AsyncSummaryHandler
from llm_mediator_simulation.utils.types import Intervention


class AsyncDebaterHandler:
    """A class to simulate the same debater across multiple debates, asynchronously"""

    def __init__(
        self,
        *,
        model: AsyncLanguageModel,
        config: DebaterConfig,
        debate_config: DebateConfig,
        summary_handler: AsyncSummaryHandler,
        parallel_debates: int = 1,
    ) -> None:
        """Initialize the asynchronous debater handler.

        Args:
            model: The language model to use.
            config: The debater configuration. The debater personality will evolve during the debate.
            debate_config: The debate configuration.
            summary_handler: The conversation summary handler.
        """

        self.model = model
        self.configs = [deepcopy(config) for _ in range(parallel_debates)]
        self.debate_config = debate_config
        self.summary_handler = summary_handler

    async def interventions(self, update_personality=False) -> list[Intervention]:
        """Do a debater intervention for all parallel debates, asynchronously

        Args:
            update_personality: Whether to update the debater personality based on the last messages before intervention.
        """

        # Update the debater personalities
        if update_personality:
            await async_debater_personality_update(
                model=self.model,
                debaters=self.configs,
                interventions=self.summary_handler.latest_messages,
            )

        responses, prompts = await async_debater_interventions(
            model=self.model,
            config=self.debate_config,
            summary=self.summary_handler,
            debaters=self.configs,
        )

        return [
            Intervention(
                debater=deepcopy(config),  # Freeze the debater configuration
                text=response["text"],
                prompt=prompt,
                justification=response["intervention_justification"],
                timestamp=datetime.now(),
            )
            for response, prompt, config in zip(responses, prompts, self.configs)
        ]
