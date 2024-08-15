"""Full debate simulation handler class"""

import pickle
from dataclasses import dataclass

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
from llm_mediator_simulation.utils.load_csv import load_csv_chat
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
        mediator_config: MediatorConfig | None = None,
        summary_config: SummaryConfig | None = None,
        metrics_handler: MetricsHandler | None = None,
    ) -> None:
        """Instanciate a debate simulation handler.

        Args:
            debater_model: The language model to use for debaters.
            mediator_model: The language model to use for the mediator and metrics.
            debaters: The debaters participating in the debate.
            config: The debate configuration.
            mediator_config: The mediator configuration. If None, no mediator will be used. Defaults to None.
            summary_config: The summary configuration. Defaults to None. A default config will be used.
            metrics_handler: The metrics handler to use. Defaults to None.
        """

        # Configuration
        self.config = config
        self.mediator_config = mediator_config
        self.summary_config = summary_config or SummaryConfig()

        # Models
        self.debater_model = debater_model
        self.mediator_model = mediator_model

        # Handlers
        self.summary_handler = SummaryHandler(
            model=mediator_model, config=self.summary_config
        )

        self.mediator_handler = (
            MediatorHandler(
                model=mediator_model,
                config=mediator_config,
                debate_config=config,
                summary_handler=self.summary_handler,
            )
            if mediator_config
            else None
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
        self.initial_debaters = [
            debater.snapshot_personality() for debater in self.debaters
        ]

    def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.

        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """

        for i in track(range(rounds)):
            for debater in self.debaters:

                ##############################################################
                #                    DEBATER INTERVENTION                    #
                ##############################################################

                intervention = debater.intervention(update_personality=i != 0)

                self.interventions.append(intervention)
                self.summary_handler.add_new_message(intervention)

                # If the debater did not intervene, skip to the next debater
                if not intervention.text:
                    continue

                if self.metrics_handler:
                    self.metrics_handler.inject_metrics(intervention)

                ##############################################################
                #                    MEDIATOR INTERVENTION                   #
                ##############################################################

                if not self.mediator_handler:
                    self.summary_handler.regenerate_summary()
                    continue

                intervention = self.mediator_handler.intervention()
                self.interventions.append(intervention)
                self.summary_handler.add_new_message(intervention)

                # Regenerate the summary for the next debater
                # (either way, a debater or mediator has intervened here)
                self.summary_handler.regenerate_summary()

    ###############################################################################################
    #                                        SERIALIZATION                                        #
    ###############################################################################################

    def pickle(self, path: str) -> None:
        """Serialize the debate configuration and logs to a pickle file.
        This does not include the model configuration.

        Args:
            path (str): The path to the pickle file, without file extension.
        """

        data: "DebatePickle" = DebatePickle(
            self.config,
            self.summary_config,
            self.mediator_config,
            self.initial_debaters,
            self.interventions,
        )

        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(
                data,
                file,
            )

    @staticmethod
    def unpickle(path: str) -> "DebatePickle":
        """Load a debate configuration and logs from a pickle file.

        Args:
            path (str): The path to the pickle file.
        """

        with open(path, "rb") as file:
            return pickle.load(file)

    def preload_chat(
        self, debaters: list[DebaterConfig], interventions: list[Intervention]
    ):
        """Preload a debate chat from debaters and interventions."""

        # Regenerate the summary handler
        self.summary_handler = SummaryHandler(
            model=self.mediator_model, config=self.summary_config
        )

        for intervention in interventions:
            self.summary_handler.add_new_message(intervention)
        self.summary_handler.regenerate_summary()

        # Regenerate mediator handler
        self.mediator_handler = (
            MediatorHandler(
                model=self.mediator_model,
                config=self.mediator_config,
                debate_config=self.config,
                summary_handler=self.summary_handler,
            )
            if self.mediator_config
            else None
        )

        # Regenerate debater handlers
        self.debaters = [
            DebaterHandler(
                model=self.debater_model,
                config=debater,
                debate_config=self.config,
                summary_handler=self.summary_handler,
            )
            for debater in debaters
        ]

        # Regenerate logs
        self.interventions = interventions
        self.initial_debaters = [
            debater.snapshot_personality() for debater in self.debaters
        ]

    def preload_csv_chat(self, path: str):
        """Preload a debate chat from a CSV file."""

        debaters, interventions = load_csv_chat(path)
        self.preload_chat(debaters, interventions)


@dataclass
class DebatePickle:
    """Pickled debate data"""

    config: DebateConfig
    summary_config: SummaryConfig
    mediator_config: MediatorConfig | None
    debaters: list[DebaterConfig]
    interventions: list[Intervention]
