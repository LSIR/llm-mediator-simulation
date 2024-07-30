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


class PersonalityAxis(Enum):
    """Debater personality axis, judged on a likert scale."""

    CIVILITY = ("civility", "civil", "toxic")
    POLITENESS = ("politeness", "polite", "rude")
    POLITICAL_ORIENTATION = ("political orientation", "conservative", "liberal")
    EMOTIONAL_STATE = ("emotional state", "calm", "angry")


class AxisPosition(Enum):
    """Position on a likert scale axis (FIRST, SECOND)."""

    VERY_FIRST = 0
    FIRST = 1
    NEUTRAL = 2
    SECOND = 3
    VERY_SECOND = 4


def axis_to_string(axis: PersonalityAxis, position: AxisPosition) -> str:
    """Convert a debater position on a likert scale axis to a string for prompting."""

    match position:
        case AxisPosition.VERY_FIRST:
            return f"very {axis.value[1]}"
        case AxisPosition.FIRST:
            return f"moderately {axis.value[1]}"
        case AxisPosition.NEUTRAL:
            return f"neutral {axis.value[0]}"
        case AxisPosition.SECOND:
            return f"moderately {axis.value[2]}"
        case AxisPosition.VERY_SECOND:
            return f"very {axis.value[2]}"


###################################################################################################
#                                    CONFIGURATION DATACLASSES                                    #
###################################################################################################


@dataclass
class DebateConfig(Promptable):
    """Debate simulation context class.

    Args:
        statement (str): The debate statement (an affirmation).
        context (str): The context of the debate.
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
    personalities: dict[PersonalityAxis, AxisPosition] | None = None

    @override
    def to_prompt(self) -> str:

        personalities: list[str] = []
        for personality, position in (self.personalities or {}).items():
            personalities.append(axis_to_string(personality, position))

        return f"""You are arguing {'in favor of' if self.position == DebatePosition.FOR else 'against'} the statement.
{"Write your text answer with a strong emphasis on the following characteristics: {', '.join(personalities)}." if len(personalities) > 0 else ""}
Remember that you are allowed to insult, diminish and curse the people you debate with."""


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
