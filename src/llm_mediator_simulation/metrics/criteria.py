"""Debate metrics criteria.

* [Politosphere criteria](https://github.com/MeloS7/Politosphere_overview/blob/main/politosphere/README.md)
* [Towards Argument Mining for Social Good](https://aclanthology.org/2021.acl-long.107.pdf)
"""

from enum import Enum

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from llm_mediator_simulation.utils.decorators import benchmark, retry
from llm_mediator_simulation.utils.json import (
    json_prompt,
    parse_llm_json,
    parse_llm_jsons,
)
from llm_mediator_simulation.utils.model_utils import Agreement, scale_description

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


@retry(attempts=5, verbose=True)
@benchmark(name="Argument Qualities", verbose=False)
def measure_argument_qualities(
    model: LanguageModel,
    text: str,
    argument_qualities: list[ArgumentQuality],
) -> dict[ArgumentQuality, Agreement]:
    """Measure the argument quality of the given text based on the given criteria.
    Returns an agreement score."""

    if len(argument_qualities) == 0:
        return {}

    json_format: dict[str, str] = {}

    for quality in argument_qualities:
        json_format[quality.name] = quality.value[1]

    prompt = f"""{text}

    Judge the text above based on the following qualities:

    {json_prompt(json_format)}

    Each JSON value should be on a scale from 0 to 4, where: {", ".join(scale_description())}
    """
    # TODO Add seed to metrics as well
    response = parse_llm_json(model.sample(prompt))

    parsed_response: dict[ArgumentQuality, Agreement] = {}

    for key, value in response.items():
        parsed_response[ArgumentQuality[key]] = Agreement(value)

    return parsed_response


async def async_measure_argument_qualities(
    model: AsyncLanguageModel,
    texts: list[str],
    argument_qualities: list[ArgumentQuality],
) -> list[dict[ArgumentQuality, Agreement]]:
    """Measure the argument quality of the given text based on the given criteria asynchronously."""

    if len(argument_qualities) == 0:
        return [{}] * len(texts)

    json_format: dict[str, str] = {}

    for quality in argument_qualities:
        json_format[quality.name] = quality.value[1]

    prompts: list[str] = []

    for text in texts:
        prompt = f"""{text}

        Judge the text above based on the following qualities:

        {json_prompt(json_format)}

        Each JSON value should be on a scale from 0 to 4, where: {", ".join(scale_description())}
        """
        prompts.append(prompt)

    responses, *_ = parse_llm_jsons(await model.sample(prompts))

    parsed_responses: list[dict[ArgumentQuality, Agreement]] = []

    for response in responses:
        parsed_response: dict[ArgumentQuality, Agreement] = {}

        for key, value in response.items():
            parsed_response[ArgumentQuality[key]] = Agreement(value)

        parsed_responses.append(parsed_response)

    return parsed_responses
