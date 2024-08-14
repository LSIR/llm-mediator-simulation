"""Async debate handler class"""

from copy import deepcopy

from rich.progress import track

from llm_mediator_simulation.metrics.metrics_handler import AsyncMetricsHandler
from llm_mediator_simulation.models.language_model import AsyncLanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.async_handler import AsyncDebaterHandler
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.summary.async_handler import AsyncSummaryHandler
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.utils.types import Intervention


class AsyncDebateHandler:
    """Async debate simulation handler class"""

    def __init__(
        self,
        *,
        debater_model: AsyncLanguageModel,
        mediator_model: AsyncLanguageModel,
        debaters: list[DebaterConfig],
        config: DebateConfig,
        mediator_config: MediatorConfig | None = None,
        summary_config: SummaryConfig | None = None,
        metrics_handler: AsyncMetricsHandler | None = None,
        parallel_debates: int = 1,
    ) -> None:
        """Instanciate an asynchronous debate simulation handler.

        Args:
            debater_model: The language model to use for debaters.
            mediator_model: The language model to use for the mediator and metrics.
            debaters: The debaters participating in the debate.
            config: The debate configuration.
            mediator_config: The mediator configuration. If None, no mediator will be used. Defaults to None.
            summary_config: The summary configuration. Defaults to None. A default config will be used.
            metrics_handler: The metrics handler to use. Defaults to None.
            parallel_debates: The number of parallel debates to run. Defaults to 1.
        """

        # Configuration
        self.config = config
        self.parallel_debates = parallel_debates

        # Handlers
        self.summary_handler = AsyncSummaryHandler(
            model=mediator_model,
            config=summary_config or SummaryConfig(),
            parallel_debates=parallel_debates,
        )

        # TODO : async mediator handler

        self.debaters = [
            AsyncDebaterHandler(
                model=debater_model,
                config=debater,
                debate_config=config,
                summary_handler=self.summary_handler,
                parallel_debates=parallel_debates,
            )
            for debater in debaters
        ]

        # Logs
        self.interventions: list[list[Intervention]] = [
            [] for _ in range(parallel_debates)
        ]
        self.initial_debaters = deepcopy(self.debaters)

    async def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.
        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """

        for i in track(range(rounds)):
            for debater_index in range(self.parallel_debates):
                pass
