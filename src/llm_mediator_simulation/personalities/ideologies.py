from dataclasses import dataclass
from enum import Enum

from llm_mediator_simulation.personalities.scales import Scale


@dataclass
class IssueValue:
    """Typing for the values of an issue."""

    name: str
    description: str | None = None


class Issues(Enum):
    GENERAL = IssueValue("general", None)
    ECONOMIC = IssueValue("economic", "economic issues")
    SOCIAL = IssueValue("social", "social issues")
    # SOCIETAL = IssueValue("societal", "societal issues")

    def __str__(self) -> str:
        """Return a printable version of the issue."""
        return self.value.name.capitalize()


class Ideology(Scale):
    """Ideologies for agents."""

    EXTREMELY_LIBERAL = "extremely liberal"
    LIBERAL = "liberal"
    SLIGHTLY_LIBERAL = "slightly liberal"
    MODERATE = "moderate"
    SLIGHTLY_CONSERVATIVE = "slightly conservative"
    CONSERVATIVE = "conservative"
    EXTREMELY_CONSERVATIVE = "extremely conservative"
    LIBERTARIAN = "libertarian"
    INDEPENDENT = "independent"

    def __str__(self) -> str:
        """Return a printable version of the ideology."""
        return self.value.capitalize()


class MonoAxisIdeology(Scale):
    EXTREMELY_LIBERAL = "extremely liberal"
    LIBERAL = "liberal"
    SLIGHTLY_LIBERAL = "slightly liberal"
    MODERATE = "moderate \n or libertarian \n or independent"
    SLIGHTLY_CONSERVATIVE = "slightly conservative"
    CONSERVATIVE = "conservative"
    EXTREMELY_CONSERVATIVE = "extremely conservative"

    def __str__(self) -> str:
        """Return a printable version of the ideology."""
        return self.value.capitalize()
