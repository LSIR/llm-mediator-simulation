"""Configuration for debate simulations"""

from dataclasses import dataclass
from typing import override

from llm_mediator_simulation.utils.interfaces import Promptable


@dataclass
class DebateConfig(Promptable):
    """Debate simulation context class.

    Args:
        statement (str): The debate statement (an affirmation).
        context (str): The context of the debate.
        add (str): The word used to refer to the action of adding a message to the conversation. Defaults to "send".
    """

    statement: str = ""
    context: str = "You are taking part in an online debate about the following topic:"
    add: str = "send"

    @override
    def to_prompt(self) -> str:
        return f"{self.context} {self.statement}."
