"""Asynchronous debater handler class"""

from copy import deepcopy
from datetime import datetime

from llm_mediator_simulation.models.language_model import AsyncLanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.simulation.prompt import (
    async_debater_interventions,
    async_debater_update,
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

    def variable_debater(self) -> bool:
        """Check if the debater is variable."""

        return self.configs[0].variable_topic_opinion or (
            self.configs[0].personality.variable_personality()
            if self.configs[0].personality is not None
            else False
        )  # To be safe, we should check these proporties for all debaters but let's assume they are the same for all debaters

    async def interventions(
        self, initial_intervention: bool = False, seed: int | None = None
    ) -> list[Intervention]:
        """Do a debater intervention for all parallel debates, asynchronously

        Args:
            initial_intervention: If this is the first intervention from this debater.
            seed: The seed to use for the random sampling at generation.
        """

        # Update the debater personalities
        if not (initial_intervention) and self.variable_debater():
            await async_debater_update(
                model=self.model,
                debate_statement=self.debate_config.statement,
                debaters=self.configs,
                interventions=self.summary_handler.latest_messages,
            )

        responses, prompts = await async_debater_interventions(
            model=self.model,
            config=self.debate_config,
            summary=self.summary_handler,
            debaters=self.configs,
            seed=seed,
        )

        return [
            Intervention(
                debater=deepcopy(
                    config
                ),  # Freeze the debater configuration because the personality can change
                text=response["text"],
                prompt=prompt,
                justification=response["intervention_justification"],
                timestamp=datetime.now(),
            )
            for response, prompt, config in zip(responses, prompts, self.configs)
        ]
