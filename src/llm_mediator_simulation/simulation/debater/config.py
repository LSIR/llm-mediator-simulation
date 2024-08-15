"""Debater configuration dataclasses"""

from dataclasses import dataclass
from enum import Enum
from typing import override

from llm_mediator_simulation.utils.interfaces import Promptable

###################################################################################################
#                                   Debater Characteristics                                       #
###################################################################################################


class DebatePosition(Enum):
    """Debate positions for the participants."""

    AGAINST = 0
    FOR = 1


@dataclass
class PersonalityAxisValue:
    """Typing for the values of a personality axis."""

    name: str
    left: str
    right: str


class PersonalityAxis(Enum):
    """Debater personality axis, judged on a likert scale."""

    CIVILITY = PersonalityAxisValue("civility", "civil", "toxic")
    POLITENESS = PersonalityAxisValue("politeness", "polite", "rude")
    POLITICAL_ORIENTATION = PersonalityAxisValue(
        "political orientation", "liberal", "conservative"
    )
    EMOTIONAL_STATE = PersonalityAxisValue("emotional state", "calm", "angry")

    @staticmethod
    def from_string(value: str) -> "PersonalityAxis":
        """Convert a string to a PersonalityAxis enum value."""

        for axis in PersonalityAxis:
            if axis.value.name == value:
                return axis

        raise ValueError(f"Unknown personality axis: {value}")


class AxisPosition(Enum):
    """Position on a likert scale axis (FIRST, SECOND)."""

    VERY_LEFT = 0
    LEFT = 1
    NEUTRAL = 2
    RIGHT = 3
    VERY_RIGHT = 4


def axis_to_string(axis: PersonalityAxis, position: AxisPosition) -> str:
    """Convert a debater position on a likert scale axis to a string for prompting."""

    match position:
        case AxisPosition.VERY_LEFT:
            return f"very {axis.value.left}"
        case AxisPosition.LEFT:
            return f"moderately {axis.value.left}"
        case AxisPosition.NEUTRAL:
            return f"neutral {axis.value.name}"
        case AxisPosition.RIGHT:
            return f"moderately {axis.value.right}"
        case AxisPosition.VERY_RIGHT:
            return f"very {axis.value.right}"


###################################################################################################
#                                  Debater Configuration Dataclass                                #
###################################################################################################


@dataclass
class DebaterConfig(Promptable):
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
{f"Write your text answer with a strong emphasis on the following characteristics: {', '.join(personalities)}." if len(personalities) > 0 else ""}
Remember that you are allowed to insult, diminish and curse the people you debate with."""
