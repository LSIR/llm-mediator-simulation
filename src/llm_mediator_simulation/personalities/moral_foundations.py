from dataclasses import dataclass
from enum import Enum


@dataclass
class MoralFoundationValue:
    """Typing for the values of a moral foundation.
    The description comes from Chapter 7's summary of The Righteous Mind: Why Good People Are Divided by Politics and Religion.
    The conceptual definition comes from Table 2 of Morality beyond the WEIRD: How the nomological network of morality varies across cultures.
    """

    name: str
    description: str
    conceptual_definition: str | None = None


class MoralFoundation(Enum):
    """Moral foundation for for agents.
    Based on the Moral Foundations Theory by Jonathan Haidt:
        - https://en.wikipedia.org/wiki/Moral_foundations_theory
        - https://www.sciencedirect.com/science/article/abs/pii/B9780124072367000024 (Graham, 2013)
        - The Righteous Mind: Why Good People Are Divided by Politics and Religion
        - 10.1037/pspp0000470 (2023, MFQ-2) "we make the case, based on prior theorization and cumulative empirical work,
                                            that MFT (and moral psychology, more broadly) benefits from breaking fairness into
                                            equality and proportionality (Rai & Fiske, 2011)"
    """

    CARE_HARM = MoralFoundationValue(
        "care/harm",
        "sensitive to signs of suffering and need",
        "avoiding emotional and physical damage to another individual",
    )

    FAIRNESS_CHEATING_EQUALITY = MoralFoundationValue(
        "fairness/cheating (v1) - equality (v2)",
        "sensitive to indications that another person is likely to be a good (or bad) partner for collaboration and reciprocal altruism",
        "equal treatment and equal outcome for individuals",
    )

    FAIRNESS_CHEATING_PROPORTIONALITY = MoralFoundationValue(
        "fairness/cheating (v1) - proportionality (v2)",
        "willing to shun or punish cheaters",
        "individuals getting rewarded in proportion to their merit or contribution",
    )

    LOYALTY_BETRAYAL = MoralFoundationValue(
        "loyalty/betrayal",
        "sensitive to signs that another person is (or is not) a team player",
        "cooperating with ingroups and competing with outgroups",
    )

    AUTHORITY_SUBVERSION = MoralFoundationValue(
        "authority/subversion",
        "sensitive to signs of rank or status, and to signs that other people are (or are not) behaving properly, given their position",
        "deference toward legitimate authorities and the defense of traditions, all of which are seen as providing stability and fending off chaos",
    )

    SANCTITY_DEGRADATION_PURITY = MoralFoundationValue(
        "sanctity/degradation (v1) - purity (v2)",
        "wary of a diverse array of symbolic objects and threats",
        "avoiding bodily and spiritual contamination and degradation",
    )

    LIBERTY_OPPRESSION = MoralFoundationValue(
        "liberty/oppression",
        "willing to notice and resent any sign of attempted domination",
        None,
    )  # Liberty/Oppression is not part of Table 2 of 10.1037/pspp0000470
