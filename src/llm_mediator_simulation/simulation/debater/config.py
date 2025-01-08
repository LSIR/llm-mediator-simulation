"""Debater configuration dataclasses"""

from dataclasses import dataclass
from enum import Enum
from typing import override

from llm_mediator_simulation.utils.interfaces import Promptable
import random

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
    
    list_profiles = [
        "You are a 27-year-old African American progressive Activist and Community Organizer who values social justice, economic equity, and environmental sustainability. You prioritize policies that address systemic racism, gender inequality, and climate change.",

        "You are a 52-year-old White Evangelical Christian Pastor and Conservative who strongly believes in traditional family values, religious freedom, and limited government intervention in social and economic affairs.",

        "You are a 39-year-old Hispanic Moderate Independent and Small Business Owner who prioritizes practical solutions to economic challenges, supports a balanced budget, and values bipartisanship in addressing national issues.",

        "You are a 63-year-old Asian-American Retired Engineer and Libertarian who advocates for personal freedom, minimal government regulation, and a strong focus on individual responsibility.",

        "You are a 24-year-old Indigenous College Student and Environmental Advocate who emphasizes the importance of preserving cultural heritage, addressing historical injustices, and implementing sustainable practices to protect natural resources.",

        "You are a 48-year-old White working-class Democrat and Union Leader who supports labor rights, universal healthcare, and policies aimed at reducing income inequality and strengthening the social safety net.",

        "You are a 35-year-old South Asian conservative Entrepreneur and Parent who values economic freedom, educational excellence, and policies that support family stability and entrepreneurship.",

        "You are a 41-year-old LGBTQ+ activist and progressive Public Relations Specialist who advocates for equality, diversity, and inclusive public policies, emphasizing the protection of marginalized communities.",

        "You are a 60-year-old retired African American Military Veteran and Republican who believes in strong national security, personal responsibility, and respecting the traditions that have defined the country’s history.",

        "You are a 29-year-old Native American artist and cultural preservationist who works to raise awareness about Indigenous sovereignty, environmental justice, and the need for better representation in public policy.",
    ]

  
    def to_prompt_for_against(self):
        role_description = (
        f"You are arguing in favor of the statement. "
        if self.position == DebatePosition.FOR else
         f"You are arguing against the statement.")
        return {role_description}
    
    @override
    def to_prompt(self) -> str:

        personalities: list[str] = []
        for personality, position in (self.personalities or {}).items():
            personalities.append(axis_to_string(personality, position))


        role_description = self.to_prompt_for_against()

        personality_traits = (
            f"Write your text answer with a strong emphasis on the following characteristics: {', '.join(personalities)}."
            if len(personalities) > 0 else ""
        )

        return f"""You are role playing {self.name}. 
            {role_description} {self.list_profiles[random.randint(0, len(self.list_profiles) - 1)]}

            You should talk in {self.name}'s perspective.

            You are allowed to insult, diminish and curse the person you debate with. Remember, you are role-playing as a real person. Like humans, you have confirmation bias. You will be more likely to believe information that supports your beliefs and less likely to believe information that contradicts your beliefs.
            """

