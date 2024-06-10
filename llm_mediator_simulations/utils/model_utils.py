"""Utilities for language model prompts."""

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


def measure_statement(model: LanguageModel, text: str, statement: str) -> int:
    """Scale the correctness of a statement regarding the given text on a scale from 0 to 10."""

    prompt = f"""{text}

    On a scale from 0 to 10, how much do you agree with the following statement about the text above?
    {statement}
    """

    return int(model.sample(prompt))
