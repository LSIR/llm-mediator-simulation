"""Debater configuration dataclasses"""

from dataclasses import dataclass
from enum import Enum
from typing import override

from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import Likert7AgreementLevel
from llm_mediator_simulation.utils.interfaces import Promptable

###################################################################################################
#                                   Debater Characteristics                                       #
###################################################################################################

@dataclass
class TopicOpinion:
    """Agent opinion on a topic."""

    agreement: Likert7AgreementLevel
    variable: bool = False


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
        name (str): The name of the debater.
        personality (Personality, optional): The debater's personality. Defaults to None.
        topic_opinion (TopicOpinion, optional): The debater's opinion on the debate topic. Defaults to None.
    """

    name: str
    personality: Personality | None = None
    topic_opinion: TopicOpinion | None = None

    @override
    def to_prompt(self) -> str:

        return f"""You are roleplaying this real person: 
        name: {self.name}; 
        {self.personality.to_prompt() if self.personality else ""}
        
        
        
        
        You are arguing {'in favor of' if self.position == DebatePosition.FOR else 'against'} the statement.
{f"Write your text answer with a strong emphasis on the following characteristics: {', '.join(personalities)}." if len(personalities) > 0 else ""}
Remember that you are allowed to insult, diminish and curse the people you debate with."""
