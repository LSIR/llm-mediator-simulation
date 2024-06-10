"""Debate metrics criteria.

* [Politosphere criteria](https://github.com/MeloS7/Politosphere_overview/blob/main/politosphere/README.md)
* [Towards Argument Mining for Social Good](https://aclanthology.org/2021.acl-long.107.pdf)
"""

from enum import Enum

from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.utils.model_utils import Agreement, measure_statement

###################################################################################################
#                                      METRICS DEFINITIONS                                        #
###################################################################################################


class ArgumentQuality(Enum):
    """Argument quality criteria from [Towards Argument Mining for Social Good](https://aclanthology.org/2021.acl-long.107.pdf)."""

    LOCAL_ACCEPTABILITY = (
        "Local acceptability",
        "The argument is sound and rationally worthy.",
    )

    LOCAL_SUFFICIENCY = ("Local sufficiency", "Enough premises support the claim.")

    LOCAL_RELEVANCE = "The premises are suitable to support the claims."

    EMOTIONAL_APPEAL = ("Emotional appeal", "The argumentation increases empathy.")

    APPROPRIATENESS = (
        "Appropriateness",
        "The language and amount of emotions are suitable.",
    )

    CREDIBILITY = (
        "Credibility",
        "The person who wrote this text is trustworthy (e.g. an expert).",
    )

    ARRANGEMENT = ("Arrangement", "The premises and claims are properly arranged.")

    GLOBAL_SUFFICIENCY = (
        "Global sufficiency",
        "Possible counterarguments are rebutted.",
    )

    # The following metric reauire knowledge of the debate topic to properly evaluate.
    GLOBAL_RELEVANCE = (
        "Global relevance",
        "The argument contributes to the resolution of the issue.",
    )
    CLARITY = (
        "Clarity",
        "Clear and correct language is used, the contribution is on topic.",
    )


###################################################################################################
#                                  METRICS MEASURMENT UTILITIES                                   #
###################################################################################################


def measure_argument_quality(
    model: LanguageModel,
    text: str,
    argument_quality: ArgumentQuality,
) -> Agreement:
    """Measure the argument quality of the given text based on the given criteria.
    Returns an agreement score."""

    return measure_statement(model, text, argument_quality.value[1])
