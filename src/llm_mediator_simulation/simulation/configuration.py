"""Configuration data for debate simulations"""

from dataclasses import dataclass
from enum import Enum
from typing import override

from llm_mediator_simulation.utils.interfaces import Promptable

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

    statement: str = ""
    context: str = "You are taking part in an online debate about the following topic:"

    @override
    def to_prompt(self) -> str:
        return f"{self.context} {self.statement}"


@dataclass
class Debater(Promptable):
    """Debater metadata class

    Args:
        position (DebatePosition): The position of the debater.
        personality (str | None, optional): The personality of the debater (as a list of qualifiers). Defaults to None.
    """

    name: str
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
    """

    mediator_preprompt: str = (
        "You are an expert mediator for a group chat. Your guidelines are the following:\n"
        "\n"
        "1. Clarify Messages: Ensure clear communication by asking for clarification if any message is unclear or ambiguous.\n"
        "2. Maintain Respect: Ensure a respectful atmosphere; intervene if the conversation becomes heated or disrespectful.\n"
        "3. Facilitate Turn-Taking: Ensure all participants have equal opportunities to speak and express their views.\n"
        "4. Encourage Constructive Feedback: Prompt participants to provide solutions and constructive feedback rather than focusing solely on problems.\n"
        "5. Summarize Key Points: Periodically summarize discussion points to ensure mutual understanding and agreement.\n"
        "6. Encourage Consensus and Move On: Guide the conversation towards alignment where possible. When participants seem to agree on which item "
        "is more important or if the conversation has reached a standstill, explicitly tell participants to consider moving to the next topic."
    )

    @override
    def to_prompt(self) -> str:
        return f"""{self.mediator_preprompt}"""
