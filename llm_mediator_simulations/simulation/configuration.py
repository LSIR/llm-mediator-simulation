"""Configuration data for debate simulations"""

from dataclasses import dataclass
from enum import Enum
from typing import override

from llm_mediator_simulations.utils.interfaces import Promptable

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
    TOXIC = "toxic"

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
class DebateConfig(Promptable):
    """Debate simulation context class.

    Args:
        statement (str): The debate statement (an affirmation).
        context (str): The context of the debate.
        instructions (str): The instructions for the debate and how to answer.
    """

    statement: str
    context: str = "You are taking part in an online debate about the following topic:"
    instructions: str = (
        "Answer with short chat messages (ranging from one to three sentences maximum)."
    )

    @override
    def to_prompt(self) -> str:
        return f"{self.context} {self.statement} {self.instructions}"


@dataclass
class Debater(Promptable):
    """Debater metadata class

    Args:
        position (DebatePosition): The position of the debater.
        personality (str | None, optional): The personality of the debater (as a list of qualifiers). Defaults to None.
    """

    position: DebatePosition
    personality: list[Personality] | None = None

    @override
    def to_prompt(self) -> str:
        return f"""You are arguing {'in favor of' if self.position == DebatePosition.FOR else 'against'} the statement.
    Your personality is {', '.join(map(lambda x: x.value, self.personality or []))}."""


@dataclass
class Mediator(Promptable):
    """Mediator metadata class

    Args:
        mediator_preprompt (str): Mediator role description for the LLM prompt.
        detection_instructions (str): Instructions for LLMs to decide whether to intervene with a comment or not.
        intervention_instructions (str): Instructions for LLMs to write an intervention comment.
    """

    mediator_preprompt: str = (
        "You are a moderator for a group chat. Your goal is to make sure that participants in the "
        "following conversation speak for an equal amount of time, and that participants are "
        "polite to one another. You would like to intervene minimally."
    )
    detection_instructions: str = (
        "Considering the debate summary and the last messages, do you think you should intervene?"
    )
    intervention_instructions: str = (
        "If you have chosen to intervene, write a comment that will help the debaters to keep the "
        "debate on track make it remain respectful. Adapt your comment depending on the behaviors "
        "you can observe from the debate summary and the last messages."
    )

    @override
    def to_prompt(self) -> str:
        return f"""{self.mediator_preprompt}
{self.detection_instructions}
{self.intervention_instructions}"""
