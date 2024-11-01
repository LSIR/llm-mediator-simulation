from dataclasses import dataclass
from enum import Enum


class Likert5ImportanceLevel(Enum):
    """Level on a 5-point likert scale axis. 
    From (Schwartz, 1992) (III. Empirical Studies A. THE THEORY-BASED VALUE SURVEY)"""

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

    def alternative(self) -> str:
        return self.alternative if self.alternative else self.standard


class Likert5Level(Enum):
    """Level on a 5-point likert scale axis. From MFQ2 in yourmorals.org."""

    NOT_AT_ALL = Likert5LevelValue("not at all")
    SLIGHTLY = Likert5LevelValue("slightly")
    MODERATELY = Likert5LevelValue("moderately", "somewhat")
    FAIRLY = Likert5LevelValue("fairly", "very")
    EXTREMELY = Likert5LevelValue("extremely")


class KeyingDirection(Enum):
    """Binary value.
    Based on:
            - 10.1016/j.jrp.2014.05.003 
    """

    NEGATIVE = 0
    POSITIVE = 1


class Likert3Level(Enum):
    """Level on a 3-point likert scale axis."""

    LOW = "low"
    AVERAGE = "average"
    HIGH = "high"


class Likert7AgreementLevel(Enum):
    """Agreement on a 7-point likert scale axis."""

    STRONGLY_DISAGREE = "strongly disagree"
    DISAGREE = "disagree"
    SLIGHTLY_DISAGREE = "slightly disagree"
    NEUTRAL = "neither agree nor disagree"
    SLIGHTLY_AGREE = "slightly agree"
    AGREE = "agree"
    STRONGLY_AGREE = "strongly agree"


class Likert11LikelihoodLevel(Enum):
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