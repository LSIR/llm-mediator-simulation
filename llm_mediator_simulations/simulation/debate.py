"""Online debate simulation handler class"""

from rich.progress import track

from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.configuration import (
    DebateConfig,
    DebatePosition,
    Debater,
)
from llm_mediator_simulations.simulation.prompt import (
    debater_comment,
    should_participant_intervene,
)
from llm_mediator_simulations.simulation.summary import Summary


class Debate:
    """Online debate simulation handler class"""

    def __init__(
        self,
        *,
        model: LanguageModel,
        debaters: list[Debater],
        configuration: DebateConfig,
        summary_handler: Summary | None = None,
    ) -> None:
        """Initialize the debate instance.

        Args:
            model (LanguageModel): The language model to use.
            debaters (list[Debater]): The debaters participating in the debate.
            configuration (str, optional): The context of the debate.
            summary_handler (Summary | None, optional): The summary handler to use. Defaults to None.
        """

        # Prompt context and metadata
        self.config = configuration

        # Positions
        self.prompt_for = "You are arguing in favor of the statement."
        self.prompt_against = "You are arguing against the statement."

        # Debaters
        self.debaters = debaters

        # Logs
        self.messages: list[tuple[DebatePosition, str]] = []

        self.model = model

        if summary_handler is None:
            self.summary_handler = Summary(model=model)
        else:
            self.summary_handler = summary_handler

    def run(self, rounds: int = 3) -> None:
        """Run the debate simulation for the given amount of rounds.
        The debaters will all send one message per round, in the order they are listed in the debaters list.
        """
        for _ in track(range(rounds)):
            for debater in self.debaters:

                if not self.should_participant_intervene(debater):
                    continue

                message = self.generate_debater_comment(debater)
                self.messages.append((debater.position, message))
                self.summary_handler.update_with_message(message)

    ###############################################################################################
    #                                     HELPERS & SHORTHANDS                                    #
    ###############################################################################################

    def should_participant_intervene(self, debater: Debater) -> bool:
        """Shorthand helper to decide whether a participant should intervene in the debate."""

        return should_participant_intervene(
            model=self.model,
            config=self.config,
            summary=self.summary_handler,
            debater=debater,
        )

    def generate_debater_comment(self, debater: Debater) -> str:
        """Shorthand helper to generate a comment for the debater."""

        return debater_comment(
            model=self.model,
            config=self.config,
            debater=debater,
            summary=self.summary_handler,
        )
