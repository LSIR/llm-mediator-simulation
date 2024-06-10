"""Utilities for language model prompts."""

from enum import Enum, EnumMeta

from llm_mediator_simulations.models.language_model import LanguageModel

###################################################################################################
#                                         SUMMARIZATION                                           #
###################################################################################################


def summarize_message(model: LanguageModel, message: str) -> str:
    """Generate a summary of the given message."""

    prompt = f"""{message}

    Summarize the message above.
    """

    return model.sample(prompt)


def summarize_conversation(model: LanguageModel, conversation: list[str]) -> str:
    """Generate a summary of the given conversation."""

    separator = "\n"
    prompt = f"""{separator.join(conversation)}

    Summarize the conversation above.
    """

    return model.sample(prompt)


###################################################################################################
#                                            SCALING                                              #
###################################################################################################


class Agreement(Enum):
    """Agreement levels for scaling."""

    STRONGLY_DISAGREE = 0
    DISAGREE = 1
    NEUTRAL = 2
    AGREE = 3
    STRONGLY_AGREE = 4


def measure_statement(model: LanguageModel, text: str, statement: str) -> Agreement:
    """Scale the correctness of a statement regarding the given text on a scale from 0 to 10."""

    scale_description: list[str] = []

    for member in Agreement:
        scale_description.append(f"{member.value} is {member.name}")

    prompt = f"""{text}

    On a scale from 0 to 4, {', '.join(scale_description)}, how much do you agree with the following statement about the text above?
    {statement}
    Answer with a single digit.
    """

    return Agreement(int(model.sample(prompt)))
