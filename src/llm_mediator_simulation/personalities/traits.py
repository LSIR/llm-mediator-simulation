from dataclasses import dataclass
from enum import Enum

from llm_mediator_simulation.personalities.scales import Likert3Level


@dataclass
class PersonalityTraitValue:
    """Typing for the values of a personality trait."""

    name: str
    low: str
    average: str
    high: str

    def level(self, level: Likert3Level = Likert3Level.HIGH) -> str:
        """Return the description for the given level."""

        if level == Likert3Level.LOW:
            return self.low
        elif level == Likert3Level.AVERAGE:
            return self.average
        elif level == Likert3Level.HIGH:
            return self.high
        else:
            raise ValueError(f"Invalid level: {level}")


class PersonalityTrait(Enum):
    """Big 5 Personality Trait for agents.
    Based on:
        - 10.18653/v1/2023.findings-emnlp.156 (Descriptions)
        - 10.1093/acrefore/9780190236557.013.560 - Table 1 (3-point Likert Scale and descriptions for each level for each trait)
        - https://en.wikipedia.org/wiki/Big_Five_personality_traits
            "A FFM-associated test was used by Cambridge Analytica, and was part of the 'psychographic profiling'
            controversy during the 2016 US presidential election."
    """

    OPENNESS = PersonalityTraitValue(
        "openness to experience",
        "You are down-to-earth, practical, traditional, and pretty much set in your ways.",
        "You are practical but willing to consider new ways of doing things. You seek a balance between the old and the new.",
        "You are open to new experiences. You have broad interests and are very imaginative.",
    )

    CONSCIENTIOUSNESS = PersonalityTraitValue(
        "conscientiousness",
        "You are easy-going, not very well organized, and sometimes careless. You prefer not to make plans.",
        "You are dependable and moderately well organized. You generally have clear goals but are able to set work aside.",
        "You are very conscientious and well organized. You have high standards and always strives to achieve goals.",
    )

    EXTRAVERSION = PersonalityTraitValue(
        "extraversion",
        "You are introverted, reserved, and serious. You prefer to be alone or with a few close friends.",
        "You are moderate in activity and enthusiasm. You enjoy the company of others but also values privacy.",
        "You are extraverted, outgoing, active, and high-spirited. You prefer to be around people most of the time.",
    )

    AGREEABLENESS = PersonalityTraitValue(
        "agreeableness",
        "You are hardheaded, skeptical, proud, and competitive. You tend to express anger directly.",
        "You are generally warm, trusting, and agreeable, but you can sometimes be stubborn and competitive.",
        "You are compassionate, good-natured, and eager to cooperate and avoid conflict.",
    )

    NEUROTICISM = PersonalityTraitValue(
        "neuroticism",
        "You are secure, hardy, and generally relaxed, even under stressful conditions.",
        "You are generally calm and able to deal with stress, but sometimes experiences feelings of guilt, anger, or sadness.",
        "You are sensitive, emotional, and prone to experience feelings that are upsetting.",
    )
