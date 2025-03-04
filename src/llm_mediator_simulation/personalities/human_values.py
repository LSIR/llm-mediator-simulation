from dataclasses import dataclass
from enum import Enum


@dataclass
class BasicHumanValuesValue:
    """Typing for the values of a basic human value.
    The description comes from the Schwartz Value Survey (10.1037/a0029393)
    """

    name: str
    description: str


class BasicHumanValues(Enum):
    """Basic human values for agents.
    Based on the Schwartz Value Survey:
        - 10.1016/s0065-2601(08)60281-6 (Schwartz, 1992) (III. Empirical Studies A. THE THEORY-BASED VALUE SURVEY for the likert scale)
        - 10.1037/a0029393 (Schwartz, 2012) - Table 2 (list and descriptions of values)
        - Idem in 10.9707/2307-0919.1173 (Schwartz, 2021) - Table 1
    """

    SELF_DIRECTION_THOUGHT = BasicHumanValuesValue(
        "self-direction thought", "freedom to cultivate one's own ideas and abilities"
    )

    SELF_DIRECTION_ACTION = BasicHumanValuesValue(
        "self-direction action", "freedom to determine one's own actions"
    )

    STIMULATION = BasicHumanValuesValue(
        "stimulation", "excitement, novelty, and change"
    )

    HEDONISM = BasicHumanValuesValue("hedonism", "pleasure and sensuous gratification")

    ACHIEVEMENT = BasicHumanValuesValue(
        "achievement", "success according to social standards"
    )

    POWER_DOMINANCE = BasicHumanValuesValue(
        "power dominance", "power through exercising control over people"
    )

    POWER_RESOURCES = BasicHumanValuesValue(
        "power resources", "power through control of material and social resources"
    )

    FACE = BasicHumanValuesValue(
        "face",
        "security and power through maintaining one's public image and avoiding humiliation",
    )

    SECURITY_PERSONAL = BasicHumanValuesValue(
        "security personal", "safety in one's immediate environment"
    )

    SECURITY_SOCIETAL = BasicHumanValuesValue(
        "security societal", "safety and stability in the wider society"
    )

    TRADITION = BasicHumanValuesValue(
        "tradition",
        "maintaining and preserving cultural, family, or religious traditions",
    )

    CONFORMITY_RULES = BasicHumanValuesValue(
        "conformity rules", "compliance with rules, laws, and formal obligations"
    )

    CONFORMITY_INTERPERSONAL = BasicHumanValuesValue(
        "conformity interpersonal", "avoidance of upsetting or harming other people"
    )

    HUMILITY = BasicHumanValuesValue(
        "humility", "recognizing one's insignificance in the larger scheme of things"
    )

    BENEVOLENCE_DEPENDABILITY = BasicHumanValuesValue(
        "benevolence dependability",
        "being a reliable and trustworthy member of the ingroup",
    )

    BENEVOLENCE_CARING = BasicHumanValuesValue(
        "benevolence caring", "devotion to the welfare of ingroup members"
    )

    UNIVERSALISM_CONCERN = BasicHumanValuesValue(
        "universalism concern",
        "commitment to equality, justice, and protection for all people",
    )

    UNIVERSALISM_NATURE = BasicHumanValuesValue(
        "universalism nature", "preservation of the natural environment"
    )

    UNIVERSALISM_TOLERANCE = BasicHumanValuesValue(
        "universalism tolerance",
        "acceptance and understanding of those who are different from oneself",
    )

    def __str__(self) -> str:
        """Return a printable version of the basic human value."""
        return self.value.name.capitalize()
