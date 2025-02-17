from dataclasses import dataclass
from enum import Enum


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


class Ideology(Enum):
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
