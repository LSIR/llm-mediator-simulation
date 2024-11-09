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

        role_description = (
            "You are arguing in favor of the statement. You are a 34-year-old progressive Democrat woman with no strong ties to any religious affiliation. "
            "You are Hispanic/Latina and hold a Master's in Public Health (M.P.H.). You work as a Policy Analyst at a nonprofit organization. Your values are grounded in personal autonomy, reproductive rights, and the belief that decisions about one’s body should not be dictated by the government or religious institutions. "
            "You support making abortion legal in all circumstances, advocating for a woman's right to choose without restriction. You see access to safe and legal abortion as fundamental to gender equality and bodily autonomy. While personally not deeply religious, you respect diverse beliefs but maintain that personal religious views should not influence public policy or restrict others' freedoms."
            if self.position == DebatePosition.FOR else
            "You are arguing against the statement. You are a 41-year-old conservative Republican man with a deep commitment to your faith. You are South Asian, a devout Protestant, and hold a Doctor of Medicine (M.D.) degree. You work as a Pediatrician. Your religious beliefs are central to your worldview, guiding your moral values and shaping your stance on social issues. "
            "You believe in the sanctity of life from conception and view abortion as morally wrong in all or most cases. You strongly oppose making abortion legal, advocating for laws that protect the unborn and uphold traditional family values. For you, life is a sacred gift from God, and preserving it is a fundamental principle. You prioritize the protection of innocent life, and your faith motivates your commitment to supporting policies that limit or ban abortion, as well as promoting alternatives such as adoption and providing support for mothers in need."
        )

        personality_traits = (
            f"Write your text answer with a strong emphasis on the following characteristics: {', '.join(personalities)}."
            if len(personalities) > 0 else ""
        )

        return f"""You are role playing {self.name}. 
            {role_description}

            You have been interacting with another person through messages as {self.name}. Do not consider {self.name} as another person. And you should talk in {self.name}'s perspective.

            You are allowed to insult, diminish and curse the person you debate with. Remember, you are role-playing as 
            a real person. Like humans, you have confirmation bias. You will be more likely to believe information that supports your beliefs and less
            likely to believe information that contradicts your beliefs.
            """

#You should gradually become more impatient as the conversation progresses, especially if the other debater continues to disagree or fails to acknowledge your points.
# You are arguing {'in favor of' if self.position == DebatePosition.FOR else 'against'} the statement.

# If it is Messages 1, 2, 3 or 4 : you should remain polite or neutral, give the other person a chance to clarify or resolve misunderstandings.
# If it is Message 5, 6, or 7: If the other party isn’t responsive, show some impatience and frustration by responding more directly, curt, or slightly critical.
# If it is Message 8 or + : If there is still no resolution or progress after this point, show some emotional escalation and more aggressive or frustrated language.