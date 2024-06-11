"""Online debate simulation handler class"""

from rich.progress import track

from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.configuration import (DebateConfig,
                                                               DebatePosition,
                                                               Debater)
from llm_mediator_simulations.simulation.summary import Summary
from llm_mediator_simulations.utils.model_utils import ask_closed_question


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
            # TODO: smarter alternance
            for debater in self.debaters:

                prompt = f"""{self.config.context} {self.config.statement}. {self.prompt_for if debater.position == DebatePosition.FOR else self.prompt_against}
                
                Your personality is {', '.join(map(lambda x: x.value, debater.personality or []))}.
                Here is a summary of the last exchanges (if empty, the conversation just started):
                {self.summary_handler.summary}

                Here are the last messages exchanged (you should focus your argumentation on them):
                {'\n\n'.join(self.summary_handler.latest_messages)}

                Do you want to add a comment to the online debate right now?
                You should often add a comment when the previous context is empty or not in the favor of your position. However, you should almost never add a comment when the previous context already supports your position.
                """

                print('------')
                print(prompt)
                print('------')

                want_to_answer = ask_closed_question(self.model, prompt)

                print(want_to_answer)
                if not want_to_answer:
                    continue

                # Prepare the prompt.
                msg_sep = "\n\n"
                prompt = f"""{self.config.context} {self.config.statement}. {self.prompt_for if debater.position == DebatePosition.FOR else self.prompt_against}
                {self.config.instructions}
                
                Your personality is {', '.join(map(lambda x: x.value, debater.personality or []))}.

                Here is a summary of the last exchanges (if empty, the conversation just started):
                {self.summary_handler.summary}

                Here are the last messages exchanged (you should focus your argumentation on them):
                {msg_sep.join(self.summary_handler.latest_messages)}
                """

                message = self.model.sample(prompt)
                self.messages.append((debater.position, message))

                self.summary_handler.update_with_message(message)
