"""Configuration data for debate simulations"""

from dataclasses import dataclass
from enum import Enum

###################################################################################################
#                                              ENUMS                                              #
###################################################################################################


class DebatePosition(Enum):
    """Debate positions for the participants."""

    AGAINST = 0
    FOR = 1


class Personality(Enum):
    """Debater personality qualifiers."""

    # Mood
    ANGRY = "angry"
    AGGRESSIVE = "aggressive"
    CALM = "calm"
    INSULTING = "insulting"
    EMPATHETIC = "empathetic"
    EMOTIONAL = "emotional"

    # Style
    REDDIT = "reddit"
    TWITTER = "twitter"
    FORMAL = "formal"
    INFORMAL = "informal"

    # Political
    CONSERVATIVE = "conservative"
    LIBERAL = "liberal"
    LIBERTARIAN = "libertarian"


###################################################################################################
#                                    CONFIGURATION DATACLASSES                                    #
###################################################################################################


@dataclass
class DebateConfig:
    """Debate simulation context class.

    Args:
        statement (str): The debate statement (an affirmation).
        context (str): The context of the debate.
        instructions (str): The instructions for the debate and how to answer.
    """

    statement: str
    context: str = "You are taking part in an online debate about the following topic:"
    instructions: str = (
        "Answer with short chat messages (ranging from one to three sentences maximum). You must convince the general public of your position."
    )


@dataclass
class Debater:
    """Debater metadata class

    Args:
        position (DebatePosition): The position of the debater.
        personality (str | None, optional): The personality of the debater (as a list of qualifiers). Defaults to None.
    """

    position: DebatePosition
    personality: list[Personality] | None = None
