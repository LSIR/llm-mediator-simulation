"""Utilities for language model prompts."""

from enum import Enum

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)

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


def summarize_conversation_with_last_messages(
    model: LanguageModel, previous_summary: str, latest_messages: list[str]
) -> str:
    """Generate a summary of the given conversation, with an emphasis on the latest messages."""

    separator = "\n\n"
    prompt = f"""Conversation summary: {previous_summary}

    Latest messages:
    {separator.join(latest_messages)}

    Summarize the conversation above, with an emphasis on the latest messages.
    """

    return model.sample(prompt)


async def summarize_conversation_with_last_messages_async(
    model: AsyncLanguageModel,
    previous_summaries: list[str],
    latest_messages: list[list[str]],
) -> list[str]:
    """Generate summaries of the given conversations, with an emphasis on the latest messages, asynchronously."""

    separator = "\n\n"
    prompts: list[str] = []

    for previous_summary, messages in zip(previous_summaries, latest_messages):
        prompt = f"""Conversation summary: {previous_summary}

        Latest messages:
        {separator.join(messages)}

        Summarize the conversation above, with an emphasis on the latest messages.
        """
        prompts.append(prompt)

    return await model.sample(prompts)


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


def scale_description() -> list[str]:
    """Return the scale description for agreement levels."""

    scale_description: list[str] = []

    for member in Agreement:
        scale_description.append(f"{member.value} is {member.name}")

    return scale_description


def measure_statement(model: LanguageModel, text: str, statement: str) -> Agreement:
    """Scale the correctness of a statement regarding the given text on a scale from 0 to 10."""

    prompt = f"""{text}

    On a scale from 0 to 4, {', '.join(scale_description())}, how much do you agree with the following statement about the text above?
    {statement}
    Answer with a single digit.
    """

    return Agreement(int(model.sample(prompt)))


###################################################################################################
#                                        CLOSED QUESTIONS                                         #
###################################################################################################


def ask_closed_question(model: LanguageModel, question: str) -> bool:
    """Ask a closed question to the model and return the answer as a boolean (yes/no)."""

    prompt = f"""{question}

    Answer the question above with "0" for no and "1" for yes.
    You must answer exactly with 0 or 1, nothing more.
    """
    return model.sample(prompt) == "1"
