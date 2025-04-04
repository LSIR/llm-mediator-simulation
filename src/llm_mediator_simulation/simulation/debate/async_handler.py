"""Async debate handler class"""

import pickle
import random
from copy import deepcopy

from rich.progress import track

from llm_mediator_simulation.metrics.async_metrics_handler import AsyncMetricsHandler
from llm_mediator_simulation.models.language_model import AsyncLanguageModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebatePickle
from llm_mediator_simulation.simulation.debater.async_handler import AsyncDebaterHandler
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.simulation.mediator.async_handler import (
    AsyncMediatorHandler,
)
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.summary.async_handler import AsyncSummaryHandler
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.utils.debaters import remove_statement_from_personalities
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
        seed: int | None = None,
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
            seed: The seed to use for the random sampling at generation. Defaults to None.
        """

        # Configuration
        self.config = config
        self.summary_config = summary_config or SummaryConfig()
        self.mediator_config = mediator_config
        self.parallel_debates = parallel_debates

        # Handlers
        self.summary_handler = AsyncSummaryHandler(
            model=mediator_model,
            config=self.summary_config,
            parallel_debates=parallel_debates,
        )

        self.mediator_handler = (
            AsyncMediatorHandler(
                model=mediator_model,
                config=mediator_config,
                debate_config=config,
                summary_handler=self.summary_handler,
            )
            if mediator_config
            else None
        )

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

        remove_statement_from_personalities(self.debaters, self.config.statement)

        self.metrics_handler = metrics_handler

        # Logs
        self.interventions: list[list[Intervention]] = [
            [] for _ in range(parallel_debates)
        ]
        self.initial_debaters = [
            deepcopy(debater.configs[0]) for debater in self.debaters
        ]

        # Seed
        self.seed = seed  # setting the seed for sampling in generation

    async def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.
        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """

        for i in track(range(rounds)):
            # Moving the internal random state initialization to the beginning of each round
            # rather than before the round loop is fairly inelegant,
            # but it enables better reproducibility through consistancy in the async case, where,
            # for each round, the order of debaters for the first parallel debate would to be the same
            # as the order of debaters in the sync case...
            if self.seed is not None:  # type: ignore[comparison-overlap]
                random.seed(
                    self.seed + i
                )  # shuffling the list of debaters consulted in each round

            # ###### Shuffle the debaters order ######
            # Say there are 2 debaters and 3 parallel debates.
            # self.debaters = [AsyncDebaterHandler(...
            #                       self.configs = [DebaterConfig(...Alice...),
            #                                       DebaterConfig(...Alice...),
            #                                       DebaterConfig(...Alice...)]
            #                       ...),
            #                  AsyncDebaterHandler(...
            #                       self.configs = [DebaterConfig(...Bob...),
            #                                       DebaterConfig(...Bob...),
            #                                       DebaterConfig(...Bob...)]
            #                       ...)]
            #
            # After shuffling, we want to have:
            # self.debaters = [AsyncDebaterHandler(...
            #                       self.configs = [DebaterConfig(...Bob...),
            #                                       DebaterConfig(...Alice...),
            #                                       DebaterConfig(...Bob...)]
            #                       ...),
            #                  AsyncDebaterHandler(...
            #                       self.configs = [DebaterConfig(...Alice...),
            #                                       DebaterConfig(...Bob...),
            #                                       DebaterConfig(...Alice...)]
            #                       ...)]
            # This is not straightforward but it matches the async calls of the debaters

            for j in range(self.parallel_debates):
                shuffled_debaters = random.sample(
                    [debater.configs[j] for debater in self.debaters],
                    len(self.debaters),
                )
                for k, debater in enumerate(self.debaters):
                    debater.configs[j] = shuffled_debaters[k]

            for debater in self.debaters:
                ######################################################################
                #                        DEBATER INTERVENTION                        #
                ######################################################################

                interventions = await debater.interventions(
                    initial_intervention=i == 0,
                    seed=self.seed,
                )
                valid_indexes = [  # Compute the indexes of debates that just had a non-empty text intervention
                    i
                    for i, intervention in enumerate(interventions)
                    if intervention.text
                ]

                if self.metrics_handler:
                    await self.metrics_handler.inject_metrics(
                        interventions, valid_indexes, self.seed
                    )

                self.append_interventions(interventions)
                self.summary_handler.add_new_messages(interventions)

                ######################################################################
                #                        MEDIATOR INTERVENTION                       #
                ######################################################################

                if not self.mediator_handler:
                    await self.summary_handler.regenerate_summaries(seed=self.seed)
                    continue

                interventions = await self.mediator_handler.interventions(
                    valid_indexes, seed=self.seed
                )

                # Mediator interventions were only computed for debates where the debater intervened, so we must pass `valid_indexes` this time
                self.append_interventions(interventions, valid_indexes)
                self.summary_handler.add_new_messages(interventions)
                await self.summary_handler.regenerate_summaries(seed=self.seed)

    def append_interventions(
        self, interventions: list[Intervention], valid_indexes: list[int] | None = None
    ) -> None:
        """Append a list of parallel interventions to their respective debates.
        If valid indexes is not None, only the debates at these indexes will be updated.
        """

        if valid_indexes is None:
            valid_indexes = list(range(self.parallel_debates))

        assert len(interventions) == len(
            valid_indexes
        ), "Interventions and valid indexes must have the same length."

        for i, intervention in zip(valid_indexes, interventions):
            self.interventions[i].append(intervention)

    def to_first_debate_pickle(self) -> DebatePickle:
        """Return the first debate configuration and logs as a DebatePickle object."""
        return DebatePickle(
            self.config,
            self.summary_config,
            self.mediator_config,
            self.initial_debaters,
            self.interventions[0],
        )

    def pickle(self, path: str) -> None:
        """Serialize all parallel debate configurations and logs to individual pickle files per debate. This does not include the model configuration.

        Args:
            path (str): The path to the pickle files, without file extension.
        """

        for i, interventions in enumerate(self.interventions):
            data = DebatePickle(
                self.config,
                self.summary_config,
                self.mediator_config,
                self.initial_debaters,
                interventions,
            )

            with open(f"{path}_{i}.pkl", "wb") as f:
                pickle.dump(data, f)
