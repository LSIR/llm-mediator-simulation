"""Full debate simulation handler class"""

import functools
import pickle
import random
from dataclasses import dataclass

from rich.progress import track

from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler
from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
    PrintableDebaterConfig,
)
from llm_mediator_simulation.simulation.debater.handler import DebaterHandler
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.mediator.handler import MediatorHandler
from llm_mediator_simulation.simulation.summary.config import (
    PrintableSummaryConfig,
    SummaryConfig,
)
from llm_mediator_simulation.simulation.summary.handler import SummaryHandler
from llm_mediator_simulation.utils.debaters import remove_statement_from_personalities
from llm_mediator_simulation.utils.load_csv import (
    load_deliberate_lab_csv_chat,
    load_reddit_csv_conv,
)
from llm_mediator_simulation.utils.types import Intervention, PrintableIntervention


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
        seed: int | None = None,
        json_debater_reponse: bool = True,
        few_shot_samples: list[dict] | None = None,
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
            seed: The seed to use for the random sampling at generation. Defaults to None.
            json_debater_reponse: Whether to enforce JSON generation for debater responses. Defaults to True.
            few_shot_samples: The few-shot samples to use for the debater. Defaults to None.
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

        remove_statement_from_personalities(self.debaters, self.config.statement)

        self.metrics_handler = metrics_handler

        # Logs
        self.interventions: list[Intervention] = []
        self.initial_debaters = [
            debater.snapshot_personality() for debater in self.debaters
        ]

        # Seed

        self.seed = seed  # setting the seed for sampling in generation

        self.debater_order: list[str] | None = None

        # JSON generation
        self.json_debater_reponse = json_debater_reponse

        # Few-shot samples
        self.few_shot_samples = few_shot_samples

    def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.

        The debaters will all send one intervention per round, in random order.
        """

        for i in track(range(rounds)):
            # Moving the internal random state initialization to the beginning of each round
            # rather than before the round loop is fairly inelegant,
            # but it enables better reproducibility through consistancy in the async case, where,
            # for each round, the order of debaters for the first parallel debate would to be the same
            # as the order of debaters in the sync case...
            if self.seed is not None:
                random.seed(
                    self.seed + i
                )  # shuffling the list of debaters consulted in each round

            if self.debater_order:
                # Follow the forced order of debaters. The first round
                debaters = []
                for debater_name in self.debater_order:
                    for debater in self.debaters:
                        if debater.config.name == debater_name:
                            debaters.append(debater)
                            break
                # If the debater is not found, raise an error
                if len(debaters) != len(self.debater_order):
                    raise ValueError(
                        f"Debater {self.debater_order[len(debaters)]} not found in the list of debaters."
                    )
                # after the first round we remove the forced order and continue the debate as usual, queriyng the debaters in round-robin
                self.debater_order = None
            else:
                # Shuffle the debaters order
                debaters = random.sample(self.debaters, len(self.debaters))

            for debater in debaters:
                ##############################################################
                #                    DEBATER INTERVENTION                    #
                ##############################################################

                intervention = debater.intervention(
                    initial_intervention=i == 0,
                    seed=self.seed,
                    json=self.json_debater_reponse,
                    few_shot_samples=self.few_shot_samples,
                )
                self.interventions.append(intervention)
                self.summary_handler.add_new_message(intervention)

                # If the debater did not intervene, skip to the next debater
                if not intervention.text:
                    continue

                if self.metrics_handler:
                    self.metrics_handler.inject_metrics(intervention, seed=self.seed)

                ##############################################################
                #                    MEDIATOR INTERVENTION                   #
                ##############################################################

                if not self.mediator_handler:
                    self.summary_handler.regenerate_summary(seed=self.seed)
                    continue

                intervention = self.mediator_handler.intervention(seed=self.seed)
                self.interventions.append(intervention)
                self.summary_handler.add_new_message(intervention)

                # Regenerate the summary for the next debater
                # (either way, a debater or mediator has intervened here)
                self.summary_handler.regenerate_summary(seed=self.seed)

    ###############################################################################################
    #                                        SERIALIZATION                                        #
    ###############################################################################################

    # To DebatePickle
    def to_debate_pickle(self) -> "DebatePickle":
        """Return the debate configuration and logs as a DebatePickle object."""

        return DebatePickle(
            self.config,
            self.summary_config,
            self.mediator_config,
            self.initial_debaters,
            self.interventions,
        )

    def pickle(self, path: str) -> None:
        """Serialize the debate configuration and logs to a pickle file.
        This does not include the model configuration.

        Args:
            path (str): The path to the pickle file, without file extension.
        """

        data: "DebatePickle" = self.to_debate_pickle()

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

        if interventions:
            self.summary_handler.regenerate_summary()
        else:
            self.summary_handler.summary = ""

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

    def preload_csv_chat(
        self,
        path: str,
        app: str = "deliberate-lab",
        truncated_num: int | None = 2,
        force_truncated_order: bool | None = None,
        load_debater_profiles: bool = False,
        debater_profiles_path: str | None = None,
        prune_debaters: bool = True,
    ):
        """Preload a debate chat from a CSV file."""
        if force_truncated_order is None:
            force_truncated_order = bool(truncated_num)

        if app == "deliberate-lab":
            load_csv_chat = load_deliberate_lab_csv_chat
        elif app == "reddit":
            load_csv_chat = functools.partial(
                load_reddit_csv_conv,
                truncated_num=truncated_num,
                force_truncated_order=force_truncated_order,
                load_debater_profiles=load_debater_profiles,
                debater_profiles_path=debater_profiles_path,
                statement=self.config.statement,
                prune_debaters=prune_debaters,
            )
        else:
            raise ValueError(
                f"Unknown app {app}. Supported apps are: 'deliberate-lab', 'reddit'."
            )

        debaters, interventions, debater_order = load_csv_chat(path)
        self.debater_order = debater_order

        self.preload_chat(debaters, interventions)


@dataclass
class PrintableDebatePikle:
    """Simpler/Printable debate data"""

    config: DebateConfig
    summary_config: PrintableSummaryConfig
    mediator_config: MediatorConfig | None
    debaters: list[PrintableDebaterConfig]
    interventions: list[PrintableIntervention]


@dataclass
class DebatePickle:
    """Pickled debate data"""

    config: DebateConfig
    summary_config: SummaryConfig
    mediator_config: MediatorConfig | None
    debaters: list[DebaterConfig]
    interventions: list[Intervention]

    def to_printable(self) -> PrintableDebatePikle:
        """Return a simpler version of the debate pickle for printing with pprint without overwhelming informations."""

        return PrintableDebatePikle(
            config=self.config,
            summary_config=self.summary_config.to_printable(),
            mediator_config=self.mediator_config,
            debaters=[debater.to_printable() for debater in self.debaters],
            interventions=[
                intervention.to_printable() for intervention in self.interventions
            ],
        )
