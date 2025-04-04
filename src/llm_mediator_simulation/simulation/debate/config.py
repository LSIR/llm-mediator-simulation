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
    """

    statement: str = ""
    context = "You are taking part in an online debate about the following topic:"

    @override
    def to_prompt(self) -> str:
        return f"{self.context} {self.statement}"
