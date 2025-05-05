from dataclasses import dataclass
from enum import Enum


class Scale(Enum):
    """Base class for scales."""

    def __str__(self) -> str:
        """Return a printable version of the scale."""
        if isinstance(self.value, str):
            return self.value.capitalize()
        else:  # isinstance(self.value, Likert5LevelValue)
            return self.value.standard.capitalize()


class Likert5ImportanceLevel(Scale):
    """Level on a 5-point likert scale axis.
    Based on "Universals in the Content and Structure of Values: Theoretical Advances and Empirical Tests in 20 Countries"
    """  # (III. Empirical Studies A. THE THEORY-BASED VALUE SURVEY)

    OPPOSED_TO_MY_VALUES = "opposed to my values"
    NOT_IMPORTANT = "not important"
    IMPORTANT = "important"
    VERY_IMPORTANT = "very important"
    OF_SUPREME_IMPORTANCE = "of supreme importance"


@dataclass
class Likert5LevelValue:
    """Typing for the values of a 5-level likert scale."""

    standard: str
    alternative: str | None = None

    def get_alternative(self) -> str:
        return self.alternative if self.alternative else self.standard


class Likert5Level(Scale):
    """Level on a 5-point likert scale axis. Based on the Moral Foundation Questionnaire-2 in yourmorals.org."""

    NOT_AT_ALL = Likert5LevelValue("not at all")
    SLIGHTLY = Likert5LevelValue("slightly")
    MODERATELY = Likert5LevelValue("moderately", "somewhat")
    FAIRLY = Likert5LevelValue("fairly", "very")
    EXTREMELY = Likert5LevelValue("extremely")


class KeyingDirection(Scale):
    """Binary value.
    Based on "Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory: Development of the IPIP-NEO-120"
    """

    NEGATIVE = "no"
    POSITIVE = "yes"


class Likert3Level(Scale):
    """Level on a 3-point likert scale axis."""

    LOW = "low"
    AVERAGE = "average"
    HIGH = "high"


class Likert7AgreementLevel(Scale):
    """Agreement on a 7-point likert scale axis."""

    STRONGLY_DISAGREE = "strongly disagree"
    DISAGREE = "disagree"
    SLIGHTLY_DISAGREE = "slightly disagree"
    NEUTRAL = "neither agree nor disagree"
    SLIGHTLY_AGREE = "slightly agree"
    AGREE = "agree"
    STRONGLY_AGREE = "strongly agree"


class Likert11LikelihoodLevel(Scale):
    """Likelihood on a 11-point likert scale axis."""

    CERTAINLY_FALSE = "you believe it is certainly false that"
    EXTREMELY_UNLIKELY = "you believe it is extremely unlikely that"
    VERY_UNLIKELY = "you believe it is very unlikely that"
    UNLIKELY = "you believe it is unlikely that"
    SOMEWHAT_UNLIKELY = "you believe it is somewhat unlikely that"
    NEUTRAL = "you do not know whether"
    SOMEWHAT_LIKELY = "you believe it is somewhat likely that"
    LIKELY = "you believe it is likely that"
    VERY_LIKELY = "you believe it is very likely that"
    EXTREMELY_LIKELY = "you believe it is extremely likely that"
    CERTAINLY_TRUE = "you believe it is certainly true that"
