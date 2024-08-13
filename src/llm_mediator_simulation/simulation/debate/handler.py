"""Full debate simulation handler class"""

from rich.progress import track

from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.simulation.debater.handler import DebaterHandler
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.mediator.handler import MediatorHandler
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.simulation.summary.handler import SummaryHandler
from llm_mediator_simulation.utils.types import Intervention


class DebateHandler:
    """Debate simulation handler class"""

    def __init__(
        self,
        *,
        debater_model: LanguageModel,
        mediator_model: LanguageModel,
        debaters: list[DebaterConfig],
        config: DebateConfig,
        mediator_config: MediatorConfig,
        summary_config: SummaryConfig | None = None,
        metrics_handler: MetricsHandler | None = None,
    ) -> None:
        """Instanciate a debate simulation handler.

        Args:
            debater_model: The language model to use for debaters.
            mediator_model: The language model to use for the mediator and metrics.
            debaters: The debaters participating in the debate.
            config: The debate configuration.
            mediator_config: The mediator configuration.
            metrics_handler: The metrics handler to use. Defaults to None.
        """

        # Configuration
        self.config = config

        # Handlers
        self.summary_handler = SummaryHandler(
            model=mediator_model, config=summary_config or SummaryConfig()
        )

        self.mediator_handler = MediatorHandler(
            model=mediator_model,
            config=mediator_config,
            debate_config=config,
            summary_handler=self.summary_handler,
        )

        self.debaters = [
            DebaterHandler(
                model=debater_model,
                config=debater,
                debate_config=config,
                summary_handler=self.summary_handler,
            )
            for debater in debaters
        ]

        self.metrics_handler = metrics_handler

        # Logs
        self.interventions: list[Intervention] = []

    def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.

        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """

        for i in track(range(rounds)):
            for debater in self.debaters:

                intervention = debater.intervention(update_personality=i != 0)

                # If the debater did not intervene, skip to the next debater
                if not intervention.text:
                    continue

                if self.metrics_handler:
                    self.metrics_handler.inject_metrics(intervention)

                self.interventions.append(intervention)
                self.summary_handler.add_new_message(intervention)

                # TODO: mediator intervention
