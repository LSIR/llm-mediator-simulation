"""Online debate simulation handler class in asynchronous mode / with batching support."""

import pickle
from datetime import datetime

from rich.progress import track

from llm_mediator_simulation.metrics.metrics_handler import AsyncMetricsHandler
from llm_mediator_simulation.models.language_model import AsyncLanguageModel
from llm_mediator_simulation.simulation.configuration import (
    DebateConfig,
    Debater,
    Mediator,
)
from llm_mediator_simulation.simulation.debate import DebatePickle
from llm_mediator_simulation.simulation.prompt import (
    async_debater_interventions,
    async_mediator_interventions,
)
from llm_mediator_simulation.simulation.summary_handler import AsyncSummaryHandler
from llm_mediator_simulation.utils.types import Intervention, LLMMessage


class AsyncDebate:
    """Online debate simulation handler class with batching support."""

    def __init__(
        self,
        *,
        debater_model: AsyncLanguageModel,
        mediator_model: AsyncLanguageModel,
        debaters: list[Debater],
        configuration: DebateConfig,
        summary_handler: AsyncSummaryHandler | None = None,
        metrics_handler: AsyncMetricsHandler | None = None,
        mediator: Mediator | None = None,
        parallel_debates: int = 1,
    ) -> None:
        """Initialize the debate instance.

        Args:
            debater_model: The language model to use for debaters.
            mediator_model: The language model to use for the mediator.
            debaters: The debaters participating in the debate.
            configuration: The context of the debate.
            summary_handler: The summary handler to use. Defaults to None.
            metrics_handler: The metrics handler to use to compute message metrics. Defaults to None.
            parallel_debates: The number of parallel debates to run. Defaults to 1.
        """

        # Prompt context and metadata
        self.config = configuration

        # Positions
        self.prompt_for = "You are arguing in favor of the statement."
        self.prompt_against = "You are arguing against the statement."

        # Debater
        self.debaters = debaters
        self.mediator = mediator
        self.parallel_debates = parallel_debates

        # Conversation detailed logs
        self.interventions: list[list[Intervention]] = [
            [] for _ in range(parallel_debates)
        ]

        self.debater_model = debater_model
        self.mediator_model = mediator_model
        self.metrics_handler = metrics_handler

        if summary_handler is None:
            self.summary_handler = AsyncSummaryHandler(
                model=mediator_model, debaters=debaters
            )
        else:
            self.summary_handler = summary_handler

    async def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.
        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """
        for _ in track(range(rounds)):
            for debater_index in range(len(self.debaters)):

                ###################################################################################
                #                                Debater intervention                             #
                ###################################################################################

                llm_response, prompts = await self.debater_interventions(
                    self.debaters[debater_index]
                )
                interventions: list[Intervention] = []

                # Add intervention to every debate, and extract the indices of the debates that need to be updated
                active_debates: list[int] = []
                for i, intervention in enumerate(llm_response):

                    if intervention["do_intervene"]:
                        active_debates.append(i)

                # Compute the metrics if a handler was provided
                if self.metrics_handler is not None:
                    # Compute metrics for the debates that were just updated
                    messages = [llm_response[i]["text"] for i in active_debates]

                    metrics = await self.metrics_handler.compute_metrics(messages)

                    interventions = [
                        Intervention(
                            debater_index,
                            llm_response[i]["text"],
                            prompts[i],
                            llm_response[i]["intervention_justification"],
                            datetime.now(),
                            metrics.pop(0) if i in active_debates else None,
                        )
                        for i in range(len(llm_response))
                    ]

                else:
                    interventions = [
                        Intervention(
                            debater_index,
                            intervention["text"],
                            prompt,
                            intervention["intervention_justification"],
                            datetime.now(),
                        )
                        for intervention, prompt in zip(llm_response, prompts)
                    ]

                for i, intervention in zip(active_debates, interventions):
                    self.interventions[i].append(intervention)

                # Add the interventions to the summary handler without regenerating the summary
                self.summary_handler.add_new_message(interventions, active_debates)

                ###################################################################################
                #                               Mediator intervention                             #
                ###################################################################################

                if self.mediator is None:
                    await self.summary_handler.regenerate_summaries(active_debates)
                    continue

                llm_response, prompts = await self.mediator_interventions()
                interventions = []

                # Add intervention to every debate, and extract the indices of the debates that need to be updated
                active_debates = []
                for i, intervention in enumerate(llm_response):

                    if intervention["do_intervene"]:
                        active_debates.append(i)

                    interventions.append(
                        Intervention(
                            None,
                            None,
                            prompts[i],
                            intervention["intervention_justification"],
                            datetime.now(),
                        )
                    )

                # Update the summaries and regenerate them
                self.summary_handler.add_new_message(interventions, active_debates)
                await self.summary_handler.regenerate_summaries(active_debates)

    ###############################################################################################
    #                                     HELPERS & SHORTHANDS                                    #
    ###############################################################################################

    async def debater_interventions(
        self, debater: Debater
    ) -> tuple[list[LLMMessage], list[str]]:
        """Shorthand helper to generate debater interventions for all debaters, asynchronously."""

        return await async_debater_interventions(
            model=self.debater_model,
            config=self.config,
            summary=self.summary_handler,
            debaters=[debater] * self.parallel_debates,
        )

    async def mediator_interventions(self) -> tuple[list[LLMMessage], list[str]]:
        """Shorthand helper to generate mediator interventions, asynchronously."""

        assert (
            self.mediator is not None
        ), "Trying to generate a mediator comment without mediator config"

        return await async_mediator_interventions(
            model=self.mediator_model,
            config=self.config,
            mediator=self.mediator,
            summary=self.summary_handler,
        )

    ###############################################################################################
    #                                        SERIALIZATION                                        #
    ###############################################################################################

    def pickle(self, path: str) -> None:
        """Serialize all debate configurations and logs to a pickle file.
        This does not include the model, summary handler, and other non data-relevant attributes.
        This function creates one file per individual debate.

        Args:
            path (str): The path to the pickle file, without a file extension.
        """

        for index, debate_interventions in enumerate(self.interventions):

            data: "DebatePickle" = DebatePickle(
                self.config, self.debaters, debate_interventions
            )

            with open(f"{path}_{index}.pkl", "wb") as file:
                pickle.dump(
                    data,
                    file,
                )
